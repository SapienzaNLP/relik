from langchain_experimental.graph_transformers import RelikGraphTransformer
from langchain_core.documents import Document

relik = RelikGraphTransformer("relik-ie/relik-relation-extraction-small-wikipedia")

text = """When Noah Lyles put his spike into Stade de France’s purple track for his first stride Sunday night of the Paris Olympics 100-meter final, he was already behind. In an event in which margin for error is slimmest, his reaction time to the starting gun was the slowest. 

Halfway through, Lyles, 27, of the U.S., was still in seventh place in an eight-man field, trying to chase down Jamaica’s Kishane Thompson, who owned not only this season’s fastest time but also the fastest time in the semifinal round contested earlier Sunday. 

By the final steps Lyles had caught up so much to Thompson, American Fred Kerley and South Africa’s Akani Simbine that he did something he rarely practices — dipping his shoulder at the finish.

Even then, Lyles was unconvinced he had won the gold medal he had so boldly predicted, and so badly wanted, for three years. The scoreboard offered no indication of who had won gold, silver or bronze as it processed a photo finish, a sold-out, raucous stadium sharing in the uncertainty.

“I think you got that one, big dog,” Lyles told Thompson. 

“I’m not even sure,” Thompson replied. “It was that close.”"""

documents = [Document(page_content=text)]
output = relik.convert_to_graph_documents(documents)
# triplets
print(output)
