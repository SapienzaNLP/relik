import argparse
from relik.common.log import get_logger
from relik.retriever import GoldenRetriever
from relik.retriever.data.datasets import AidaInBatchNegativesDataset
from relik.retriever.indexers.document import DocumentStore
from relik.retriever.indexers.inmemory import InMemoryDocumentIndex
from relik.retriever.trainer import RetrieverTrainer

logger = get_logger(__name__)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("encoder", type=str, required=True)
    arg_parser.add_argument("index", type=str, required=True)
    args = arg_parser.parse_args()

    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=args.encoder,
        document_index=args.index,
        device="cuda",
    )

    # val_dataset = AidaInBatchNegativesDataset(
    #     name="aida_val",
    #     path="/root/relik-sapienzanlp/data/retriever/el/aida_32_tokens_topic_relik/val.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     use_topics=True,
    # )
    test_dataset = AidaInBatchNegativesDataset(
        name="aida_test",
        path="/root/relik-sapienzanlp/data/retriever/el/aida_32_tokens_topic_relik/test.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=True,
    )

    trainer = RetrieverTrainer(
        retriever=retriever,
        # train_dataset=train_dataset,
        # val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        max_steps=25_000,
        log_to_wandb=False,
        # wandb_online_mode=False,
        # wandb_project_name="relik-retriever-aida",
        # wandb_experiment_name="aida-e5-base-topics-from-blink-new-data",
        max_hard_negatives_to_mine=15,
        resume_from_checkpoint_path=None,  # path to lightning checkpoint
        trainer_kwargs={"logger": False},
    )

    # trainer.train()
    trainer.test()
