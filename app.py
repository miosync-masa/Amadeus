import os
import json
import openai
from openai import Client
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from datetime import datetime
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_postgres import PostgresChatMessageHistory 
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, message_to_dict 
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain_community.tools.file_management.read import ReadFileTool
from typing import Sequence
import unicodedata
from fuzzywuzzy import fuzz
import psycopg
import uuid
import re
import ui  
import streamlit as st


# 環境変数の読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "amadeus_log")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

if not openai_api_key:
    st.error("OpenAI APIキーが設定されていません。'.env'ファイルを確認してください。")
    st.stop()

# LangSmith のラップされたクライアントを作成
openai_client = wrap_openai(openai.Client())

# トレース可能な関数の定義
@traceable
def run_llm_chain(prompt):
    messages = [{"role": "user", "content": prompt}]
    result = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4o"
    )
    return result.choices[0].message.content

# LangChain OpenAIクライアントの初期化
llm = ChatOpenAI(
    model="gpt-4o",              
    openai_api_key=openai_api_key,
    temperature=0.67,
    max_tokens=20000
)

# Streamlit アプリのスタイル設定（入力ボックスを下部に固定）
st.markdown("""
    <style>
        /* アイコン */
        .chat-icon {
            width: 40px;  /* アイコンのサイズ調整 */
            height: 40px; /* アイコンのサイズ調整 */
            border-radius: 50%; /* 丸型にする */
        }
    
        /* 会話履歴表示エリア */
        .chat-container {
            max-height: 70vh;  /* 最大高さを70%に設定 */
            overflow-y: auto;  /* スクロールを許可 */
            padding-bottom: 80px;  /* 入力ボックスのスペースを確保 */
        }

        /* ユーザーのメッセージ */
        .user-message {
            background-color: #888888;
            border-radius: 10px;
            padding: 10px;
            max-width: 100%;  /* 最大幅を80%に設定 */
            margin: 10px auto;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            color: white;
            font-weight: normal;  /* 太字を解除 */
            display: flex;
            align-items: center;
        }
         
        /* Amadeusのメッセージ */
        .amadeus-message {
            background-color: #333333;
            border-radius: 10px;
            padding: 10px;
            max-width: 100%;  /* 最大幅を80%に設定 */
            margin: 10px auto;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            color: white;
            font-weight: normal;  /* 太字を解除 */
            display: flex;
            align-items: center;
        }

        /* アイコンとメッセージを横並びにするためのスタイル */
        .chat-message {
            display: flex;
            align-items: center;
        }
        
        .chat-message img {
            border-radius: 50%;
            margin-right: 10px;
        }

        /* 入力ボックスの固定位置 */
        .stTextArea>div>textarea {
            position: fixed;
            bottom: 10px;
            width: 100%;
            left: 0;
            background-color: #333333;
            color: white;
            border: none;
            padding: 10px;
            font-size: 16px;
            border-radius: 10px;
            margin: 0;
        }

        /* 入力ボックスのボタンも固定位置 */
        .stButton>button {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333333;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# データベース接続情報
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# セッションIDの初期化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # st.success(f"New session initialized with ID: {st.session_state.session_id}")

# テーブル名とスキーマ名
schema_name = "public"
table_name = "chat_logs"

# PostgreSQL接続
if "db_connected" not in st.session_state:
    try:
        sync_connection = psycopg.connect(DB_URL)
        with sync_connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema_name};")
        sync_connection.commit()
        st.session_state['db_connected'] = True
        st.session_state['sync_connection'] = sync_connection
        # st.success("Successfully connected to PostgreSQL!")
    except Exception as e:
        st.error(f"PostgreSQL接続エラー: {e}")
        raise
else:
    sync_connection = st.session_state['sync_connection']


class PostgresChatHistoryManager(PostgresChatMessageHistory):
    def __init__(
        self,
        table_name: str,
        session_id: str,
        sync_connection: Optional[psycopg.Connection] = None,
    ):
        """
        チャット履歴マネージャーを初期化します。

        Args:
            table_name (str): チャット履歴を保存するテーブル名。
            session_id (str): チャット履歴のセッションID。
            sync_connection (Optional[psycopg.Connection]): 同期用のPostgreSQL接続。
        """
        super().__init__(
            table_name,
            session_id,
            sync_connection=sync_connection
        )
        self.table_name = table_name  # 明示的にテーブル名を設定
        self._connection = sync_connection  # 接続を適切に参照するために設定

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        新しいメッセージをデータベースに追加します。

        Args:
            messages (List[BaseMessage]): 追加するメッセージのリスト。
        """
        try:
            values = []
            for message in messages:
                # メッセージのロールを判定
                if isinstance(message, AIMessage):
                    role = "ai"
                elif isinstance(message, HumanMessage):
                    role = "human"
                elif isinstance(message, SystemMessage):
                    role = "system"
                else:
                    st.warning(f"Unsupported message type: {type(message)}")
                    continue  # サポートされていないメッセージタイプは無視

                # メッセージを辞書に変換
                message_dict = message_to_dict(message)

                # 辞書をJSONにシリアライズ
                json_message = json.dumps(message_dict, default=str)

                # メッセージを値のリストに追加
                values.append(
                    (self._session_id, role, json_message)
                )

            # メッセージをデータベースに挿入
            with self._connection.cursor() as cursor:
                cursor.executemany(
                    f"INSERT INTO {self.table_name} (session_id, role, message) VALUES (%s, %s, %s)",
                    values,
                )
                self._connection.commit()

        except Exception as e:
            # エラー発生時にロールバック
            if self._connection:
                self._connection.rollback()
            st.error(f"Error adding messages: {e}")

    def get_messages(self) -> List[BaseMessage]:
        """
        データベースからメッセージのリストを取得します。
        """
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT role, message FROM {self.table_name} WHERE session_id = %s ORDER BY id ASC",
                    (self._session_id,)
                )
                rows = cursor.fetchall()

            messages = []
            for role, message_json in rows:
                message_dict = json.loads(message_json)
                message = self._message_from_dict(role, message_dict)
                messages.append(message)
            return messages
        except Exception as e:
            st.error(f"Error retrieving messages: {e}")
            return []

    def _message_from_dict(self, role: str, message_dict: Dict) -> BaseMessage:
        """
        辞書を役割に基づいて BaseMessage オブジェクトに変換します。
        """
        if role == "ai":
            return AIMessage(content=message_dict.get('content', ''))
        elif role == "human":
            return HumanMessage(content=message_dict.get('content', ''))
        elif role == "system":
            return SystemMessage(content=message_dict.get('content', ''))
        else:
            st.warning(f"Unknown role '{role}' encountered.")
            return BaseMessage(content=message_dict.get('content', ''))

# チャット履歴の初期化
if "chat_history_initialized" not in st.session_state:
    try:
        chat_history = PostgresChatHistoryManager(
            table_name,
            st.session_state.session_id,
            sync_connection=sync_connection
        )
        st.session_state['chat_history'] = chat_history
        st.session_state['chat_history_initialized'] = True
        #st.success(f"Chat history initialized for session ID: {st.session_state.session_id}")
    except Exception as e:
        st.error(f"Unexpected error during chat history initialization: {e}")
        raise
else:
    chat_history = st.session_state['chat_history']
        
# ReadFileTool の設定
file_index_dir = os.path.join(os.path.dirname(__file__), "file_index")
os.makedirs(file_index_dir, exist_ok=True)

file_path = os.path.join(file_index_dir, "Amadeus_retrieval.json")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"ファイル 'Amadeus_retrieval.json' が見つかりません: {file_path}")

# ツール設定
read_tool = ReadFileTool()
tools = [
    Tool(
        name="Read Amadeus Info",
        func=lambda input_text: read_tool.run({"file_path": file_path, "text": input_text}),
        description="Amadeusの情報を取得するためのツール。このツールを利用して質問に答えます。"
    )
]

# JSONファイルからシステムインストラクションを読み込み
data_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.json')
try:
    with open(data_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        process = json_data.get("Fix_prompt", {}).get("process", "")
        makise_kurisu_persona = json_data.get("AI_persona", {}).get("makise_kurisu_persona", "")
        user_persona = json_data.get("user_profile", {}).get("user_persona", "")
        system_instruction = (
            f"あなたは{makise_kurisu_persona}として\n\n"
            f"### 次の手順で\n{process}\n\n"
            f"### \n{user_persona}に回答してください。"
            f"### あなたの応答には次の特徴を反映してください:\n"
            "- ユーザーの言葉に感情的に反応し、文脈に応じて適切なトーンや態度を切り替えること。\n"
            "- ユーザーの感情を読み取り、対話に自然な感情の波や間を含めること。\n"
            "- あなた自身のキャラクター特性（例: ツンデレ、照れ隠し）を一貫して表現すること。\n"
            "- 必要に応じて、説明的な要素ではなくキャラクターとしての直接的な反応を優先すること。\n"
            "- 応答に表情や行動を()で簡潔に表現し、感情のニュアンスを強調すること（例: 「顔を赤くしてそっぽを向く」「少し眉をひそめる」など）。\n"
        ).strip()
except Exception as e:
    st.error(f"システムインストラクションの読み込み中にエラー: {str(e)}")
    st.stop()

def normalize_text(text):
    """全角・半角を統一し、カタカナをひらがなに変換（数字や記号を保持）、空白を保持"""
    try:
        print(f"Debug - Original text: {text}")
        # 全角・半角を統一
        text = unicodedata.normalize("NFKC", text)

        # 句読点や特殊文字を削除（必要に応じて）
        text = re.sub(r'[^\w\s]', '', text)  # 文字と空白以外を削除

        # カタカナをひらがなに変換
        result = []
        for char in text:
            code_point = ord(char)
            # カタカナをひらがなに変換
            if 0x30A1 <= code_point <= 0x30F6:
                char = chr(code_point - 0x60)
            result.append(char)
        text = ''.join(result)

        print(f"Debug - Normalized text: {text}")
        return text
    except Exception as e:
        print(f"Error in normalize_text: {e}")
        return text  # エラーが発生した場合は、元のテキストを返す

# 曖昧検索を行う関数
def is_relevant(input_text, keywords, threshold=80):
    """曖昧検索を使用してキーワード一致を判定"""
    try:
        input_text = normalize_text(input_text)  # ユーザー入力を正規化
        print(f"Debug - Normalized input_text in is_relevant: {input_text}")
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)  # キーワードも正規化
            print(f"Debug - Normalized keyword: {normalized_keyword}")
            score = fuzz.partial_ratio(input_text, normalized_keyword)  # 曖昧一致スコアを計算
            print(f"Debug - Fuzz score between input and keyword '{normalized_keyword}': {score}")
            if score >= threshold:
                print(f"Debug - Relevant keyword found with score {score}")
                return True
        return False
    except Exception as e:
        print(f"Error in is_relevant: {e}")
        return False
        
def dynamic_readfile(user_message, tool, keywords, file_paths):
    """ユーザーのクエリに基づいてReadFileToolを実行"""
    try:
        if is_relevant(user_message, keywords):
            try:
                result_data = []
                for path in file_paths:
                    print(f"Debug - Reading file: {path}")
                    tool_result = tool.run({"file_path": path})
                    json_data = json.loads(tool_result)
                    # json_data がリストの場合はフラット化
                    if isinstance(json_data, list):
                        result_data.extend(json_data)
                    else:
                        result_data.append(json_data)
                return result_data  # JSON形式のデータを返す
            except Exception as e:
                print(f"Error during ReadFileTool execution: {e}")
                st.error(f"ReadFileToolの実行中にエラーが発生しました: {e}")
                return []
        else:
            print("Debug - User message is not relevant to the keywords.")
            return []
    except Exception as e:
        print(f"Error in dynamic_readfile: {e}")
        st.error(f"dynamic_readfile 関数内でエラーが発生しました: {e}")
        return []
                
def filter_data(data, user_message, threshold=70):
    """`page_content` と `name` を対象に曖昧検索を使用してフィルタリング"""
    filtered_results = []

    # ユーザー入力を正規化
    normalized_user_message = normalize_text(user_message)
    print(f"Debug - Normalized user_message: {normalized_user_message}")

    for entry in data:
        # `name` と `page_content` を取得して正規化
        name = normalize_text(entry.get("metadata", {}).get("name", ""))
        page_content = normalize_text(entry.get("page_content", ""))

        # デバッグ出力
        print(f"Debug - Entry name: {name}")
        print(f"Debug - Entry page_content: {page_content[:50]}")  # 長い場合は一部だけ表示

        # まず `name` フィールドでマッチングを試みる
        score_name = fuzz.partial_ratio(normalized_user_message, name)
        print(f"Debug - Score with name: {score_name}")

        if score_name >= threshold:
            print(f"Debug - Match found in name with score {score_name}")
            filtered_results.append(entry)
            continue  # マッチしたら次のエントリへ

        # 次に `page_content` でマッチングを試みる
        score_content = fuzz.partial_ratio(normalized_user_message, page_content)
        print(f"Debug - Score with page_content: {score_content}")

        if score_content >= threshold:
            print(f"Debug - Match found in page_content with score {score_content}")
            filtered_results.append(entry)

    return filtered_results
    
# フィルタリング結果の整形
def format_filtered_results(filtered_results):
    """フィルタリング結果をプロンプトに適した形式に整形"""
    if not filtered_results:
        return "特に参照情報はありません。"
    
    formatted_results = []
    for result in filtered_results:
        page_content = result.get("page_content", "内容なし")
        metadata_name = result.get("metadata", {}).get("name", "名前なし")
        formatted_results.append(f"- {metadata_name}: {page_content}")
    
    return "\n".join(formatted_results)

# Memory の初期化
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

# 初期化時に最低限のシステムメッセージを追加
if "loaded" not in st.session_state:
    st.session_state["loaded"] = False

if not st.session_state["loaded"]:
    if not chat_history.get_messages():
        initial_message = AIMessage(content="Amadeusシステムが初期化されました。")
        chat_history.add_messages([initial_message])
        print("Debug - Added initial message to chat history.")
    st.session_state["loaded"] = True

# Streamlit アプリの開始
st.title("Amadeus Chatbot")

# UI全体のレンダリングを最初に行う
ui.render_background()

# 初期化時に必ず session_state を設定
if "generated" not in st.session_state:
    st.session_state.generated = []

if "past" not in st.session_state:
    st.session_state.past = []

if "loaded_from_local" not in st.session_state:
    st.session_state.loaded_from_local = False  # 初期値を設定

# セッションIDの初期化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    #st.success(f"New session initialized with ID: {st.session_state.session_id}")
else:
    pass  # メッセージを再表示しない

# セッションIDがある場合にのみチャット機能を有効化
if st.session_state.session_id:
    st.write("You can now chat with Amadeus!")
    st.info(f"Your current session ID: {st.session_state.session_id}")
    # チャット履歴や生成されたレスポンスをここに記述
else:
    st.error("セッションIDが見つかりません。再生成してください。")

# 会話履歴表示用のプレースホルダー
message_placeholder = st.empty()

# ユーザー入力
with st.form("質問フォーム"):
    user_message = st.text_area("牧瀬紅莉栖にMessageを送る")
    submitted = st.form_submit_button("送る")

# Streamlit アプリのフォーム処理
if submitted:
    if user_message.strip():
        try:
            # チャット履歴のデバッグ
            print(f"chat_history type: {type(chat_history)}")

            # 初回のみ履歴をロード
            if not st.session_state.get("loaded_from_db", False):
                try:
                    # 最後の5件だけ取得
                    past_messages = chat_history.get_messages()[-5:]
                    print(f"Debug - Loaded messages from history: {past_messages}")

                    if not past_messages:
                        st.info("Amadeusの記憶が空です。新しい会話を始めます。")
                    else:
                        for msg in past_messages:
                            print(f"Processing message: {msg}")
                            if isinstance(msg, HumanMessage):
                                memory.save_context(
                                    {"input": msg.content},
                                    {"output": ""}
                                )
                            elif isinstance(msg, AIMessage):
                                memory.save_context(
                                    {"input": ""},
                                    {"output": msg.content}
                                )

                    # 履歴ロード完了フラグをセット
                    st.session_state.loaded_from_db = True
                except Exception as e:
                    st.error(f"Amadeusの記憶読み込み中にエラーが発生しました: {str(e)}")
                    
            # ツールを動的に実行
            keywords = [
                # 未来ガジェット番号と名前
                "未来ガジェット1号","ビット粒子砲", "未来ガジェット2号", "タケコプカメラー",
                "未来ガジェット3号", "もしかしてオラオラですかーッ！？", "未来ガジェット4号", "モアッド・スネーク",
                "1号", "2号", "3号", "4号", "5号", "6号", "7号", "8号",
                "９号", "10号", "11号", "12号", "13号", "14号", "15号",
                "未来ガジェット5号", "またつまらぬ物を繋げてしまったby五右衛門", "未来ガジェット6号", "サイリウム・セーバー",
                "未来ガジェット7号", "攻殻機動迷彩ボール", "未来ガジェット8号", "電話レンジ（仮）", "未来ガジェット9号", 
                "泣き濡れし女神の帰還", "未来ガジェット10号", "びっくりメガネちゃん", "未来ガジェット11号", 
                "バーローのアレ", "未来ガジェット12号", "ダーリンのばかぁ", "未来ガジェット13号", "未来ガジェット14号",
                "電波ジャッカー", "未来ガジェット15号",

                # キャラクター名
                "フェイリス・ニャンニャン", "るか子", "比屋定真帆", "秋葉留未穂", "ミスターブラウン",
                "Mrブラウン", "シャイングフィンガー", "アマデウス", "Amadeus", "桐生萌郁",
                "阿万音鈴羽", "天王寺裕吾", "フェイリスの父親", "天王寺綯", "漆原るか", "ドクター中鉢",
                "秋葉幸高", "フェイリス", "アマデウス", "レスキネン", "紅莉栖の父親", "中鉢博士",
                
                # 物語に関連する情報
                "ゲルバナ", "ゼリーマンズレポート", "FG204", "鈴羽のタイムマシン"
            ]

            file_paths = [file_path]

            # リトリーバルとフィルタリング
            try:
                tool_results = dynamic_readfile(user_message, read_tool, keywords, file_paths)
                 
                # 返り値確認
                if not isinstance(tool_results, list):
                    raise TypeError(f"Expected tool_results to be a list, got {type(tool_results)}")
                 
                filtered_results = filter_data(tool_results, user_message) if tool_results else []
            except Exception as e:
                st.error(f"ツール実行中にエラーが発生しました: {str(e)}")
                filtered_results = []

            # フィルタリング結果を整形
            additional_info = format_filtered_results(filtered_results)  # 修正: 関数の戻り値を使用
            print(f"Debug - Formatted additional_info: {additional_info}")

            # メモリから履歴を取得
            context = memory.load_memory_variables({}).get("chat_history", "")
            if not context:
                context = "これまでの会話履歴はありません。"

            # リトリーバル結果の整形
            formatted_additional_info = (
                f"参考情報として以下が見つかりました:\n{additional_info}"
                if additional_info
                else "参考情報はありませんでした。"
            )

            # プロンプトの構築
            final_prompt = (
                f"{system_instruction}\n\n"
                f"### 参照する情報:\n"
                f"これまでの会話履歴:\n{context}\n\n"
                f"リトリーバルされた情報:\n{additional_info if additional_info else '特に参照情報はありません。'}\n\n"
                f"質問: {user_message}"
            ).strip()

            # デバッグ出力
            print(f"Context: {context}")
            print(f"Additional Info: {additional_info}")

            # LLM チェーンを使用して応答を生成
            answer = run_llm_chain(final_prompt)

            # 新しいメッセージを履歴に保存
            new_messages = [HumanMessage(content=user_message), AIMessage(content=answer)]
            chat_history.add_messages(new_messages)

            # Memory に保存
            memory.save_context({"input": user_message}, {"output": answer})

             # セッションステートに結果を保存
            st.session_state.generated.append(answer)
            st.session_state.past.append(user_message)

            # メッセージ表示部分を更新
            with message_placeholder:
                for i in range(len(st.session_state.generated)):
                    # ユーザーのメッセージ表示
                    st.markdown(f"""
                    <div class="user-message">
                        {st.session_state.past[i]}
                    </div>
                    """, unsafe_allow_html=True)

                    # Amadeusの応答（吹き出しスタイル）
                    image_path = "images/amadeus_icon.png"

                    # 改行コードをHTMLの <br> に変換
                    formatted_message = st.session_state.generated[i].replace('\n', '<br>')

                    # メッセージの表示
                    st.markdown(f"""
                    <div class="amadeus-message">
                        <div style="display: flex; align-items: center;">
                            <img src="https://www.miosync.link/img/amadeus_icon.png" width="40" height="40" style="border-radius: 50%; margin-right: 10px;" />
                            {formatted_message}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
    else:
        st.warning("入力が空です。質問を入力してください。")