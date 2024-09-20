import sys
import os

import logging
import datetime
import click

from typing import Annotated, List, TypedDict
import operator

import sqlalchemy
from connect_connector import connect_with_connector
from connect_connector_auto_iam_authn import connect_with_connector_auto_iam_authn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger()

db = None

GOOGLE_AI=os.environ.get("GOOGLE_AI")

# LLM初期化
if GOOGLE_AI == "GEMINI":
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
else:
    llm = VertexAI(model="gemini-1.5-flash")
    #llm = VertexAI(model="gemini-1.0-pro")

# 会話者の初期化
SPEAKERS = []
SPEAKERS_NAMES = []

def init_connection_pool() -> sqlalchemy.engine.base.Engine:
    return (
        connect_with_connector_auto_iam_authn()
        if os.environ.get("DB_IAM_USER")
        else connect_with_connector()
    )

    raise ValueError(
        "Missing database connection type. Please define one of INSTANCE_CONNECTION_NAME"
    )

class AppState(TypedDict):
    """
    アプリケーション実行中に保持される状態を表すクラス
    
    Attributes:
      thema(str): 会話のテーマ
      history(List[str]): 会話の履歴
      speak_count(int): 全体の発言回数
      next_speaker(str): 次の発言者の名前
      conversation_summary(str): 会話全体の要約
    """
    thema: str
    history: Annotated[List[str], operator.add]
    speak_count: int
    next_speaker: str
    conversation_summary: str

def speaker(state: AppState):
    """
    会話の参加者として発言を生成します。

    Args:
      state(AppState): AppState
    Return:
      Dict[str]: 生成した発言
    """
    speaker_name = state.get("next_speaker",None)
    history = state.get("history",[])
    thema = state.get("thema","")
    speak_count = state.get("speak_count",0)
    model = llm
    
    system_message = f"""あなたは次のようなパーソナリティを持った人物です。
    ------------
    {SPEAKERS[SPEAKERS_NAMES.index(speaker_name)]}
    ------------
    この人物"{speaker_name}"として他の参加者と{thema}について会話をし、全員で結論を出してください。
    """

    human_message_prefix = f"""あなたは今{thema}について他の参加者と会話をし、全員で結論を導くタスクが与えられています。
    これまでの会話の履歴を見て、あなたの{thema}についての意見を自然な短い文体で作成してください。
    あなたから誰かの意見を仰ぐ発言はしてはいけません。

    # 会話の履歴
    """
    human_message = human_message_prefix + "\n".join(history) + f"\n{speaker_name}: "

    response = model.invoke(
        [
            SystemMessage(content=system_message), 
            HumanMessage(content=human_message_prefix)
        ]
    )
    print(str(response))
    response_msg = str(response.content) # Gemini API
    #response_msg = str(response) # VertexAI
    return {
        "history" : [f"{speaker_name}: {response_msg}"], 
        "speak_count" : speak_count + 1
    }

def conversation_manager(state: AppState):
    """
    会話の管理者として次の発言者を決定します。

    Args:
      state(AppState): AppState
    Return:
      Dict[str]: 次の発言者の名前
    """
    max_speak_count = 9 # 会話の回数の最大値
    first_speaker_index = 0 # 一番最初の発言者のindex
    speak_count = state.get("speak_count",0)
    thema = state.get("thema","")
    history = state.get("history",[])
    model = llm
    
    greeting_msg = f"""司会: 今日は[{thema}]についてのみなさんの活発なご意見を聞かせて下さい！"""
    speakers_names_str = ",".join(SPEAKERS_NAMES)
    
    if speak_count > max_speak_count:
        # 会話の最大回数まで到達した場合は強制終了
        return {"next_speaker": "no_one"}

    if speak_count == 0:
        # 会話の開始時はConversationManagerによる挨拶を送り、index=0の発言者を指名する
        return {"history": [greeting_msg], "next_speaker": SPEAKERS_NAMES[first_speaker_index]}

    system_message = f"""{speakers_names_str}が{thema}についての会話をしています。
    あなたはこの会話を管理する役割を持っています。
    与えられる会話の履歴を読み、次に発言すべき参加者の名前を決定します。"""
    
    human_message = f"""これまでの{thema}についての[{speakers_names_str}]の履歴を見て、
    会話の結論がまとまるまで次に誰が発言すべきかを決めて下さい。
    必ず一人2回以上発言させるようにしてください。
    同じような内容の会話が連続しこれ以上会話に変化が見られないと判断した場合は[TERMINATE]と出力してください。
    それ以外は必ず[{speakers_names_str}]のどれかを出力してください。

    # 履歴
    {history}

    次の発言者: 
    """

    response = model.invoke(
        [
            SystemMessage(content=system_message), 
            HumanMessage(content=human_message)
        ]
    )

    # 出力テキストのパース処理
    print(response)
    if GOOGLE_AI == "GEMINI":
        generate_text = response.content.replace("次の発言者:","").strip().replace("'","").replace("*","") # GeminiAPI
    else:
        generate_text = response.replace("次の発言者:","").strip().replace("'","").replace("*","") # VertexAI TODO うまくSpeakerが取れていない。

    #print(generate_text)　※Geminiの場合　**B太**のように出てくるので、**を除去している。実際はもう少し良い方法で抽出した方が良い。

    if generate_text in SPEAKERS_NAMES:
        # 次の発言者が指定された場合
        return {"next_speaker": generate_text}
    elif "TERMINATE" in generate_text:
        # 会話の終了と判断された場合
        return {"next_speaker": "no_one"}
    else:
        # それ以外
        return {"next_speaker": None}

def conversation_summarizer(state: AppState):
    """
    会話の最後に会話の内容を要約し、インサイトを作成します。

    Args:
      state(AppState)
    Return:
      Dict[str]: 生成した会話から得られるインサイト
    """
    thema = state.get("thema","")
    history = state.get("history",[])
    model = llm
    
    system_message = "あなたは複数人の会話から有益なインサイトを見つけることに長けています。"
    human_message = f"""これまでの{thema}についての会話の履歴を見てください。
    そのあと、その会話からどんなインサイトが得られるかを考え、出力してください。
    出力はできるだけ人が読みやすい形式で100文字程度で出力してください。

    # 履歴
    {history}
    
    """
    response = model.invoke(
        [
            SystemMessage(content=system_message), 
            HumanMessage(content=human_message)
        ]
    )
    
    if GOOGLE_AI == "GEMINI":
        return{ "conversation_summary": response.content } # Gemini API
    else:
        return{"conversation_summary": response} # VertexAI

def next_speaker(state: AppState):
    """
    次の発言者を決めます。
    
    Args:
      state: AppState
    Return:
      str: 次のアクション
    """
    speaker_name = state.get("next_speaker",None)

    if speaker_name in SPEAKERS_NAMES:
        # 次の発言者が指定された場合
        return "continue"
    elif speaker_name == "no_one":
        # 会話が完了した場合
         return "summarize"
    else:
        # それ以外
        return "end"

def main():
  
    global db
    if db is None:
        db = init_connection_pool()

    stmt = sqlalchemy.text('SELECT * FROM meeting where start_date is null order by meeting_id')
    try:
        with db.connect() as conn:
            res = conn.execute(stmt, parameters={})
            meetings = res.fetchall()
    except Exception as e:
        logger.exception(e)

    for meeting in meetings:

        # AI Meeting Start
        start_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        query = """
        UPDATE meeting
           SET start_date = :start_date,
               meeting_status = :meeting_status,
               updated_date = :updated_date
         WHERE meeting_id = :meeting_id
        """

        stmt = sqlalchemy.text(query)
        try:
            with db.connect() as conn:
                conn.execute(stmt, parameters={"meeting_id": meeting.meeting_id, "start_date": start_date, "meeting_status": 2, "updated_date": start_date})
                conn.commit()
        except Exception as e:
            logger.exception(e)

        query2 = """
            SELECT p.name, st.stereo_content
              FROM personality as p
              JOIN meeting_personality as mp on p.id = mp.id
              JOIN stereo_type as st on p.stereo_type = st.stereo_type
             Where mp.meeting_id=:meeting_id
        """
        stmt2 = sqlalchemy.text(query2)
        try:
            with db.connect() as conn:
                res = conn.execute(stmt2, parameters={"meeting_id": meeting.meeting_id})
                personalities = res.fetchall()
        except Exception as e:
            logger.exception(e) 

        global SPEAKERS
        global SPEAKERS_NAMES

        SPEAKERS.clear()
        SPEAKERS_NAMES.clear()

        for personality in personalities:
            SPEAKERS.append({personality.name : personality.stereo_content})
            SPEAKERS_NAMES.append(personality.name)

        workflow = StateGraph(AppState)
        workflow.add_node("conversation_manager", conversation_manager)
        workflow.add_node("speaker", speaker)
        workflow.add_node("conversation_summarizer", conversation_summarizer)

        workflow.add_conditional_edges(
            "conversation_manager",
            next_speaker,
            {
                "continue": "speaker",
                "summarize": "conversation_summarizer",
                "end": END
            }
        )

        workflow.add_edge("speaker", "conversation_manager")
        workflow.add_edge("conversation_summarizer", END)
        workflow.set_entry_point("conversation_manager")
        group_discussion = workflow.compile()

        summary = "" # type string
        result = "" # type string

        for event in group_discussion.stream({
            "thema": meeting.theme,
            "speak_count": 0,
            "history":[],
            "next_speaker": None,
            "conversation_summary": ""
        }):
            for value in event.values():
                if 'history' in value:
                    summary = summary + str(value['history'][0]) + "\n"
                
                if 'conversation_summary' in value:
                    print(str(value))
                    result = result + str(value['conversation_summary'])

        end_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        query = """
            UPDATE meeting
               SET result = :result,
                   summary = :summary,
                   end_date = :end_date,
                   meeting_status = :meeting_status,
                   updated_date = :updated_date
             WHERE meeting_id = :meeting_id
        """
        stmt = sqlalchemy.text(query)
        try:
            with db.connect() as conn:
                conn.execute(
                    stmt, parameters={
                        "meeting_id": meeting.meeting_id, "result": result, "summary": summary,
                        "end_date": end_date, "meeting_status": 3, "updated_date": end_date
                    }
                )
                conn.commit()
        except Exception as e:
            logger.exception(e)            

if __name__ == "__main__":
    main()

