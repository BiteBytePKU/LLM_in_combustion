import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from server import myChat

# Streamlit页面配置
st.set_page_config(page_title="AI4C", page_icon=":robot_face:")
with st.sidebar:
    st.title('燃烧领域专家模型')
    st.markdown('---')
    # 定义一个包含两个选项的列表
    options = ['minicpm', 'RAG-minicpm']

    # 创建一个下拉菜单，并将"minicpm"设置为默认选项
    # 因为"minicpm"是列表中的第一个选项，所以index为0
    selected_option = st.selectbox('请选择模型', options, index=0)

    # 根据选中的选项设置status变量的值
    status = 1 if selected_option == 'minicpm' else 2
    st.toast(f'您选择的模型是：{selected_option}')

    st.markdown('特性：\n- RAG版本具有索引增强\n- 上下文记忆')
    st.markdown('---')
    # 显示一个带有自定义属性的消息

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
    elif isinstance(message, SystemMessage):
        with st.chat_message('system'):
            st.markdown(message.content)

if user_input := st.chat_input("请输入"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)
    with st.chat_message("assistant"):
        response = myChat.get_reply(user_input, status)
        answer = response["answer"] if status != 1 else response
        docs = response["source_documents"] if status != 1 else []
        st.markdown(answer)
        st.session_state.user_input = ""
    st.session_state.messages.append(AIMessage(content=answer))
    if status != 1:
        i = 1
        for doc in docs:
            content = f"文档{i}：\n{doc.page_content}\n{str(doc.metadata)}\n"
            st.chat_message("system").markdown(content)
            st.session_state.messages.append(SystemMessage(content=content))
            i += 1

