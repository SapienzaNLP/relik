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
        name="nuner_train",
        path="/root/relik-sapienzanlp/data/retriever/el/nuner/nuner_train.150K.windowed.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        use_topics=False,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="nuner_val",
        path="/root/relik-sapienzanlp/data/retriever/el/nuner/nuner_valid.windowed.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=False,
    )
    test_dataset = AidaInBatchNegativesDataset(
        name="nuner_test",
        path="/root/relik-sapienzanlp/data/retriever/el/nuner/nuner_valid.windowed.dpr.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=False,
    )

    logger.info("Loading document index")
    document_index = InMemoryDocumentIndex(
        documents=DocumentStore.from_file("/root/relik-sapienzanlp/data/retriever/el/nuner/ner_types.jsonl"),
        metadata_fields=None,
        separator=None,
        device="cuda",
        precision="16",
    )
    retriever.document_index = document_index

    trainer = RetrieverTrainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        top_k=25,
        max_steps=150_000,
        wandb_online_mode=True,
        wandb_project_name="golden-retriever-nuner",
        wandb_experiment_name="nuner-e5-small-no-hard-neg",
        max_hard_negatives_to_mine=0,
    )

    trainer.train()
    trainer.test()
