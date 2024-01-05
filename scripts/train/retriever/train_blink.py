from relik.common.log import get_logger
from relik.retriever import GoldenRetriever
from relik.retriever.data.datasets import (
    AidaInBatchNegativesDataset,
    SubsampleStrategyEnum,
)
from relik.retriever.indexers.document import DocumentStore
from relik.retriever.indexers.inmemory import InMemoryDocumentIndex
from relik.retriever.trainer import Trainer

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(question_encoder="intfloat/e5-base-v2")

    train_dataset = AidaInBatchNegativesDataset(
        name="aida_train",
        path="/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/first_1M.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        subsample_strategy=SubsampleStrategyEnum.RANDOM,
        # use_topics=True,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="aida_val",
        path="/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/val.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        # use_topics=True,
    )
    # test_dataset = AidaInBatchNegativesDataset(
    #     name="aida_test",
    #     path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/test.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     use_topics=True,
    # )

    logger.info("Loading document index")
    document_index = InMemoryDocumentIndex(
        documents=DocumentStore.from_file(
            "/root/golden-retriever/data/entitylinking/documents.jsonl"
        ),
        metadata_fields=["definition"],
        separator=" <def> ",
        device="cuda",
        precision="16",
    )
    retriever.document_index = document_index

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        num_workers=0,
        max_steps=400_000,
        wandb_online_mode=True,
        wandb_project_name="relik-retriever-blink",
        wandb_experiment_name="blink-first1M-e5-base-topics",
        max_hard_negatives_to_mine=15,
        mine_hard_negatives_with_probability=0.2,
        save_last=True,
        resume_from_checkpoint_path=None,  # path to lightning checkpoint
    )

    trainer.train()
    trainer.test()
