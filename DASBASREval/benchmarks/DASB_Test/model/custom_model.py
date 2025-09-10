import torch


class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w


class Discrete_LLM_EmbeddingLayer(torch.nn.Module):
    """This class handles embedding layers  for discrete tokens.

    Arguments
    ---------
    num_codebooks: int ,
        number of codebooks of the tokenizer.
    vocab_size : int,
        size of the dictionary of embeddings
    emb_dim: int ,
        the size of each embedding vector
    pad_index: int (default: 0),
        If specified, the entries at padding_idx do not contribute to the gradient.
    init: boolean (default: False):
        If set to True, init the embedding with the tokenizer embedding otherwise init randomly.
    freeze: boolean (default: False)
       If True, the embedding is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.

    Example
    -------
    >>> from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> print(tokens.shape)
    torch.Size([4, 4, 2])
    >>> emb= Discrete_EmbeddingLayer(2, 1024, 1024)
    >>> in_emb = emb(tokens)
    >>> print(in_emb.shape)
    torch.Size([4, 4, 2, 1024])
    """

    def __init__(
        self,
        num_codebooks,
        vocab_size,
        llm_dim,
        emb_dim,
        pad_index=0,
        init=False,
        freeze=False,
    ):
        super(Discrete_LLM_EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.freeze = freeze
        self.embedding = torch.nn.Embedding(
            num_codebooks * vocab_size, llm_dim
        ).requires_grad_(not self.freeze)
        self.emb_dense = torch.nn.Linear(llm_dim, emb_dim)
        self.init = init

        from transformers import AutoModel, AutoTokenizer
        llm_model_name = "allenai/longformer-base-4096"

        llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.float32, device_map="cpu")
        llm_embeddings = llm_model.embeddings.word_embeddings.weight
        llm_embedding_w = llm_embeddings[4:4+num_codebooks * vocab_size]

        self.init_embedding(llm_embedding_w)

    def init_embedding(self, weights):
        with torch.no_grad():
            self.embedding.weight = torch.nn.Parameter(weights)

    def forward(self, in_tokens):
        """Computes the embedding for discrete tokens.
        a sample.

        Arguments
        ---------
        in_tokens : torch.Tensor
            A (Batch x Time x num_codebooks)
            audio sample
        Returns
        -------
        in_embs : torch.Tensor
        """

        # breakpoint()

        with torch.set_grad_enabled(not self.freeze):
            #  Add unique token IDs across diffrent codebooks by adding num_codebooks * vocab_size
            in_tokens += torch.arange(
                0,
                self.num_codebooks * self.vocab_size,
                self.vocab_size,
                device=in_tokens.device,
            )
            in_embs = self.embedding(in_tokens)
            return in_embs



class Discrete_Codebook_EmbeddingLayer(torch.nn.Module):
    """This class handles embedding layers  for discrete tokens.

    Arguments
    ---------
    num_codebooks: int ,
        number of codebooks of the tokenizer.
    vocab_size : int,
        size of the dictionary of embeddings
    emb_dim: int ,
        the size of each embedding vector
    pad_index: int (default: 0),
        If specified, the entries at padding_idx do not contribute to the gradient.
    init: boolean (default: False):
        If set to True, init the embedding with the tokenizer embedding otherwise init randomly.
    freeze: boolean (default: False)
       If True, the embedding is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.

    Example
    -------
    >>> from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> print(tokens.shape)
    torch.Size([4, 4, 2])
    >>> emb= Discrete_EmbeddingLayer(2, 1024, 1024)
    >>> in_emb = emb(tokens)
    >>> print(in_emb.shape)
    torch.Size([4, 4, 2, 1024])
    """

    def __init__(
        self,
        num_codebooks,
        vocab_size,
        ssl_emb_dim,
        emb_dim,
        codec,
        pad_index=0,
        init=False,
        freeze=False,
    ):
        super(Discrete_Codebook_EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.freeze = freeze
        self.embedding = torch.nn.Embedding(
            num_codebooks * vocab_size, ssl_emb_dim
        ).requires_grad_(not self.freeze)
        self.init = init

        codebook_weights = []

        for code_book_idx in range(num_codebooks):
            codebook_vecs_np = codec.kmeans_models[code_book_idx].cluster_centers_.astype('float32')
            codebook_vecs = torch.from_numpy(codebook_vecs_np).to(torch.float32)
            codebook_weights.append(codebook_vecs)

        codebook_weights = torch.cat(codebook_weights, dim=0)

        self.emb_dense = torch.nn.Linear(ssl_emb_dim, emb_dim)

        codec = codec.cpu()

        self.init_embedding(codebook_weights)


    def init_embedding(self, weights):
        with torch.no_grad():
            self.embedding.weight = torch.nn.Parameter(weights).requires_grad_(False)

    def forward(self, in_tokens):
        """Computes the embedding for discrete tokens.
        a sample.

        Arguments
        ---------
        in_tokens : torch.Tensor
            A (Batch x Time x num_codebooks)
            audio sample
        Returns
        -------
        in_embs : torch.Tensor
        """

        with torch.set_grad_enabled(not self.freeze):
            #  Add unique token IDs across diffrent codebooks by adding num_codebooks * vocab_size
            in_tokens += torch.arange(
                0,
                self.num_codebooks * self.vocab_size,
                self.vocab_size,
                device=in_tokens.device,
            )
            # Forward Pass to embedding and
            in_embs = self.embedding(in_tokens)
            in_embs = self.emb_dense(in_embs)
            return in_embs


class Reprogram_Layer(torch.nn.Module):
    """This class handles embedding layers  for discrete tokens.

    Arguments
    ---------
    num_codebooks: int ,
        number of codebooks of the tokenizer.
    vocab_size : int,
        size of the dictionary of embeddings
    emb_dim: int ,
        the size of each embedding vector
    pad_index: int (default: 0),
        If specified, the entries at padding_idx do not contribute to the gradient.
    init: boolean (default: False):
        If set to True, init the embedding with the tokenizer embedding otherwise init randomly.
    freeze: boolean (default: False)
       If True, the embedding is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.

    Example
    -------
    >>> from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> print(tokens.shape)
    torch.Size([4, 4, 2])
    >>> emb= Discrete_EmbeddingLayer(2, 1024, 1024)
    >>> in_emb = emb(tokens)
    >>> print(in_emb.shape)
    torch.Size([4, 4, 2, 1024])
    """

    def __init__(
        self,
        emb_dim,
        pad_index=0,
        freeze=False,
        init=False
    ):
        super(Reprogram_Layer, self).__init__()

        self.init = init
        self.freeze = freeze

        from transformers import AutoModel, AutoTokenizer
        llm_model_name = "allenai/longformer-base-4096"

        llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.float32, device_map="cpu")   

        llm_embeddings = llm_model.embeddings.word_embeddings.weight[4:6004].detach()
        self.llm_embeddings = llm_embeddings

        self.prototype_dense = torch.nn.Linear(6000, 1000)

        self.reprogram_layer = CrossAttention(emb_dim, emb_dim)



    def forward(self, in_embs):
        """ Cross-att Reprogram

        """

        with torch.set_grad_enabled(not self.freeze):
            #  Add unique token IDs across diffrent codebooks by adding num_codebooks * vocab_size

            llm_embeddings = self.prototype_dense(self.llm_embeddings.to(in_embs.device).transpose(1,0)).transpose(1,0)
            llm_embeddings = self.llm_embeddings.to(in_embs.device)
            llm_embeddings = llm_embeddings.unsqueeze(0).repeat(in_embs.size(0), 1, 1)
            reprogram_in_embs = self.reprogram_layer(in_embs, llm_embeddings, llm_embeddings)

            return reprogram_in_embs


# Define Cross-Attention Layer
class CrossAttention(torch.nn.Module):
    def __init__(self, d_model_q, d_model_k):
        super(CrossAttention, self).__init__()

        self.d_model = d_model_k

        self.W_Q = torch.nn.Linear(d_model_q, d_model_k, bias=False)
        self.W_K = torch.nn.Linear(d_model_k, d_model_k, bias=False)
        self.W_V = torch.nn.Linear(d_model_k, d_model_k, bias=False)

        self.init_weights()

    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        return attention_weights @ V

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.W_Q.weight)
        torch.nn.init.xavier_uniform_(self.W_K.weight)
        torch.nn.init.xavier_uniform_(self.W_V.weight)





