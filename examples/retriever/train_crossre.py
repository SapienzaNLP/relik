from relik.common.log import get_logger
from relik.retriever.indexers.document import DocumentStore
from relik.retriever.trainer import RetrieverTrainer
from relik.retriever import GoldenRetriever
from relik.retriever.indexers.inmemory import InMemoryDocumentIndex
from relik.retriever.data.datasets import AidaInBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="intfloat/e5-small-v2"
    )

    train_dataset = AidaInBatchNegativesDataset(
        name="crossre_train",
        path="/root/relik-sapienzanlp/data/retriever/re/crossre/train.swapped.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=256,
        passage_batch_size=6,
        max_passage_length=64,
        shuffle=True,
        use_topics=False,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="biorel_val",
        path="/root/relik-sapienzanlp/data/retriever/re/crossre/dev.swapped.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=256,
        passage_batch_size=6,
        max_passage_length=64,
        use_topics=False,
    )
    test_dataset = AidaInBatchNegativesDataset(
        name="biorel_test",
        path="/root/relik-sapienzanlp/data/retriever/re/crossre/test.swapped.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=256,
        passage_batch_size=6,
        max_passage_length=64,
        use_topics=False,
    )

    logger.info("Loading document index")
    document_index = InMemoryDocumentIndex(
        documents=DocumentStore.from_file("/root/relik-sapienzanlp/data/retriever/re/crossre/index_bio.jsonl"),
        metadata_fields=["definition"],
        separator=" <def> ",
        device="cuda",
        precision="16",
    )
    retriever.document_index = document_index

    trainer = RetrieverTrainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=0,
        max_steps=100_000,
        val_check_interval=0.2,
        wandb_online_mode=True,
        top_k=[4, 8, 12],
        wandb_project_name="golden-retriever-crossre",
        wandb_experiment_name="crossre-e5-small",
        max_hard_negatives_to_mine=5,
    )

    trainer.train()
    trainer.test()
