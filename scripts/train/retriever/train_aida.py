from relik.common.log import get_logger
from relik.retriever import GoldenRetriever
from relik.retriever.data.datasets import AidaInBatchNegativesDataset
from relik.retriever.indexers.document import DocumentStore
from relik.retriever.indexers.inmemory import InMemoryDocumentIndex
from relik.retriever.trainer import Trainer

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="/root/golden-retriever/wandb/blink-first1M-e5-base-topics/files/retriever/question_encoder",
        document_index=InMemoryDocumentIndex(
            documents=DocumentStore.from_file(
                "/root/golden-retriever/data/entitylinking/documents.jsonl"
            ),
            metadata_fields=["definition"],
            separator=" <def> ",
            device="cuda",
            precision="16",
        ),
    )

    train_dataset = AidaInBatchNegativesDataset(
        name="aida_train",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/train.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        use_topics=True,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="aida_val",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/val.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=True,
    )
    test_dataset = AidaInBatchNegativesDataset(
        name="aida_test",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/test.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=True,
    )

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        max_steps=25_000,
        wandb_online_mode=True,
        wandb_project_name="relik-retriever-aida",
        wandb_experiment_name="aida-e5-base-topics-from-blink",
        max_hard_negatives_to_mine=15,
        resume_from_checkpoint_path=None,  # path to lightning checkpoint
    )

    trainer.train()
    trainer.test()
