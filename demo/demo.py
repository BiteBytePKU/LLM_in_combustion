import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from server import myChat
# 实例化核心功能对象

# Streamlit页面配置
msg = "状态"
st.set_page_config(page_title="聊天机器人", page_icon=":robot_face:")
with st.sidebar:
    st.title('知识库与参数设置')
    st.markdown('---')
    st.markdown('这是它的特性：\n- 索引增强\n- 上下文记忆\n- 多种模式')
    st.markdown('---')
    # 显示一个带有自定义属性的消息
    # 功能待补充
    st.text(msg)

st.title('ChatBot')
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
messages = st.session_state.get('messages', [])
for message in messages:
    if isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.markdown(message.content)

if user_input := st.chat_input("请输入"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    with st.chat_message("assistant"):
        response = myChat.get_reply(user_input)
        answer = response["answer"]
        docs = response["source_documents"]
        total = answer+"\n引用文档："+str(docs)
        # response = response.replace("$", "\$")  # disable latex for $ sign
        st.markdown(answer)
        st.session_state.user_input = ""
    st.session_state.messages.append(AIMessage(content=answer))

