from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

application_prompt = """Tu dois analyser profondément le texte de l'utilisateur et lui trouver un titre. Le titre doit contenir une touche humoristique : Voici le texte
  {user_input}
"""

user_input = """ Le match entre Garry Kasparov ett Anatoly est l'une des rivalités les plus légendaires de l'histoire des échecs. Leur premier affrontement pour le titre mondial
en 1984 a été un marathon épuisant, s'étalant sur 48 parties avant d'être annulé par la FIDE. Karpov menait initialement, mais Kasparov, avec son style dynamique et agressif, a 
commencer à renverser la tendance. L'année suivante, en 1985, Kasparov a finalement pris sa revanche en battant Karpov et en devenant le plus jeune champion du monde. Leur rivalité
a continué avec plusieurs matches de championnat du monde jusqu'àen 1990, chacun offrant des parties spectaculaires et stratégiquement complexes. Karpov, maître du jeu positionnel,
contrastait avec Kasparovn, adepte du jeu tactique explosif et novateur. Leur opposition symbolisait aussi une lutte entre deux générations et deux stylesde pensée. Kasparov a fini
par dominer, mais Karpov a marqué l'histoire des échecs, captivant des millions de fans à travers le monde. Aujourd'hui encore, les parties entre ces deux légendes sont étudiées par
les amateurs et les grands maîtres du jeu"""


# Model
llm = ChatOpenAI(
    temperature=1,
    max_tokens=500,
    model='gpt-4o'
)

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=application_prompt
)


chain = prompt | llm | StrOutputParser()
result = chain.invoke({"user_input": user_input})

