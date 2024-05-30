from pprintpp import pprint

from relik.inference.annotator import Relik
from relik.inference.data.objects import TaskType
from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
from relik.retriever.pytorch_modules.model import GoldenRetriever


def main():
    # retriever = GoldenRetriever(
    #     question_encoder="riccorl/retriever-relik-entity-linking-aida-wikipedia-base-question-encoder",
    #     document_index="riccorl/retriever-relik-entity-linking-aida-wikipedia-base-index",
    #     device="cuda",
    #     index_device="cpu",
    #     precision=16,
    #     index_precision=32,
    # )
    # reader = RelikReaderForSpanExtraction(
    #     "riccorl/reader-relik-entity-linking-aida-wikipedia-small"
    # )

    # relik = Relik(
    #     retriever=retriever,
    #     reader=reader,
    #     top_k=100,
    #     window_size=32,
    #     window_stride=16,
    #     task=TaskType.SPAN,
    # )
    # relik.save_pretrained(
    #     "relik-entity-linking-aida-wikipedia-tiny",
    #     save_weights=False,
    #     push_to_hub=True,
    #     # reader_model_id="reader-relik-entity-linking-aida-wikipedia-small",
    #     # retriever_model_id="retriever-relik-entity-linking-aida-wikipedia-base",
    # )

    relik = Relik.from_pretrained(
        "/root/relik-sapienzanlp/pretrained_configs/relik-reader-base-giuliano-style-no-proj-special-token",
        device="cuda",
        precision=16,
    )

    # input_text = "But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer, the erratic former world chess champion, dished out in March at a news conference in Reykjavik, Iceland."
    input_text = "EU rejects German call to boycott British lamb . Peter Blackburn BRUSSELS 1996-08-22 The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep . Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer . \" We do n't support any such recommendation because we do n't see any grounds for it , \" the Commission 's chief spokesman Nikolaus van der Pas told a news briefing . He said further scientific study was required and if it was found that action was needed it should be taken by the European Union . He said a proposal last month by EU Farm Commissioner Franz Fischler to ban sheep brains , spleens and spinal cords from the human and animal food chains was a highly specific and precautionary move to protect human health . Fischler proposed EU-wide measures after reports from Britain and France that under laboratory conditions sheep could contract Bovine Spongiform Encephalopathy ( BSE ) -- mad cow disease . But Fischler agreed to review his proposal after the EU 's standing veterinary committee , mational animal health officials , questioned if such action was justified as there was only a slight risk to human health . Spanish Farm Minister Loyola de Palacio had earlier accused Fischler at an EU farm ministers ' meeting of causing unjustified alarm through \" dangerous generalisation . \" . Only France and Britain backed Fischler 's proposal . The EU 's scientific veterinary and multidisciplinary committees are due to re-examine the issue early next month and make recommendations to the senior veterinary officials . Sheep have long been known to contract scrapie , a brain-wasting disease similar to BSE which is believed to have been transferred to cattle through feed containing animal waste . British farmers denied on Thursday there was any danger to human health from their sheep , but expressed concern that German government advice to consumers to avoid British lamb might influence consumers across Europe . \" What we have to be extremely careful of is how other countries are going to take Germany 's lead , \" Welsh National Farmers ' Union ( NFU ) chairman John Lloyd Jones said on BBC radio . Bonn has led efforts to protect public health after consumer confidence collapsed in March after a British report suggested humans could contract an illness similar to mad cow disease by eating contaminated beef . Germany imported 47,600 sheep from Britain last year , nearly half of total imports . It brought in 4,275 tonnes of British mutton , some 10 percent of overall imports ."
    output = relik(input_text)
    print(output)
    # input_text = """
    # But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer, the erratic former world chess champion, dished out in March at a news conference in Reykjavik, Iceland.
    # """
    # input_text = "RT @username: It’s easier for teenagers in Texas to buy an AR-15 than it is a handgun, or even a beer. The high-powered AR-15 rifle, similar to the Army’s M-16, is the weapon of choice for many mass murderers bent on achieving the highest body count possible. #banAR"
    # input_text = "SOCCER - JAPAN GET LUCKY WIN\",\" CHINA IN SURPRISE DEFEAT. Nadim Ladki AL-AIN\",\" United Arab Emirates 1996-12-06 Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday. But China saw their luck desert them in the second match of the group\",\" crashing to a surprise 2-0 defeat to newcomers Uzbekistan. China controlled most of the match and saw several chances missed until the 78th minute when Uzbek striker Igor Shkvyrin took advantage of a misdirected defensive header to lob the ball over the advancing Chinese keeper and into an empty net. Oleg Shatskiku made sure of the win in injury time\",\" hitting an unstoppable left foot shot from just outside the area. The former Soviet republic was playing in an Asian Cup finals tie for the first time. Despite winning the Asian Games title two years ago\",\" Uzbekistan are in the finals as outsiders. Two goals from defensive errors in the last six minutes allowed Japan to come from behind and collect all three points from their opening meeting against Syria. Takuya Takagi scored the winner in the 88th minute\",\" rising to head a Hiroshige Yanagimoto cross towards the Syrian goal which goalkeeper Salem Bitar appeared to have covered but then allowed to slip into the net. It was the second costly blunder by Syria in four minutes. Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar's goal. Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute. Japan then laid siege to the Syrian penalty area for most of the game but rarely breached the Syrian defence. Bitar pulled off fine saves whenever they did. Japan coach Shu Kamo said : ' ' The Syrian own goal proved lucky for us. The Syrians scored early and then played defensively and adopted long balls which made it hard for us. ' ' Japan\",\" co-hosts of the World Cup in 2002 and ranked 20th in the world by FIFA\",\" are favourites to regain their title here. Hosts UAE play Kuwait and South Korea take on Indonesia on Saturday in Group A matches. All four teams are level with one point each from one game."
    # input_text = "SOCCER - JAPAN GET LUCKY WIN\",\" CHINA IN SURPRISE DEFEAT. Nadim Ladki AL-AIN\",\" United Arab Emirates 1996-12-06 Japan began the"
    # input_text = "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . Nadim Ladki AL-AIN , United Arab Emirates 1996-12-06 Japan began the"
    # preds = relik(input_text, annotation_type="word", progress_bar=True)
    # retriever.retrieve(text, k=100, batch_size=128, progress_bar=True)
    # pprint(preds)


if __name__ == "__main__":
    main()
