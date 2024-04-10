from langchain.tools.render import render_text_description

from croptalk.tools import tools

RENDERED_TOOLS = render_text_description(tools)
TOOL_PROMPT = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{RENDERED_TOOLS}
""" + """\
Given the user questions, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

Do not use tools if they are not necessary.

This is the question you are being asked : {question}

"""
