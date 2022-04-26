from modeling_bart import BartModel

class PipelinedBertForPretraining(transformers.BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBertForPretraining(config).parallelize().half().train()
        ```
        """
        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            fused = BertFusedSelfAttention(self.config)
            fused.load_state_dict(layer.attention.self.state_dict())
            layer.attention.self = fused

        if self.config.embedding_serialization_factor > 1:
            serialized_decoder = SerializedLinear(self.config.hidden_size,
                                                  self.config.vocab_size,
                                                  self.config.embedding_serialization_factor,
                                                  bias=True,
                                                  mode=poptorch.MatMulSerializationMode.OutputChannels)
            serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = serialized_decoder
            self.tie_weights()

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
                recomputation_checkpoint(layer)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")

        logger("Pooler     --> IPU 0")
        self.bert.pooler = poptorch.BeginBlock(self.bert.pooler, "Pooler", ipu_id=0)

        logger("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger("-----------------------------------------------------------")
        return self

    def _init_weights(self, module):
        """Initialize the weights"""
        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_positions, masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs[:2]

        # Select only the masked tokens for the classifier
        masked_output = self.gather_indices(sequence_output, masked_lm_positions)

        prediction_scores, sequential_relationship_score = self.cls(masked_output, pooled_output)
        outputs = (prediction_scores, sequential_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=0).float()
            next_sentence_loss = F.cross_entropy(sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

            next_sentence_acc = accuracy(sequential_relationship_score.view([-1, 2]), next_sentence_label.view(-1))
            # masked_lm_labels: 0 if corresponding token not masked, original value otherwise
            masked_lm_acc = accuracy_masked(prediction_scores.view([-1, self.config.mask_tokens, self.config.vocab_size]), masked_lm_labels, 0)
            outputs = (total_loss, masked_lm_loss, next_sentence_loss, masked_lm_acc, next_sentence_acc)

        return outputs
