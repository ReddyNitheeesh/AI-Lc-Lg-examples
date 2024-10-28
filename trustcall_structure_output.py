from typing import List, Optional

from pydantic import BaseModel


# schema

class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str


class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None


class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None


class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None


class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None


class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore


class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences


class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences


# LLM Call

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
bound = llm.with_structured_output(TelegramAndTrustFallPreferences)

conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

bound.invoke(f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>""")


from trustcall import create_extractor

bound = create_extractor(
    llm,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences",
)

result = bound.invoke(
    f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>"""
)
result["responses"][0]

# output in below format

#
# {
#     "pertinent_user_preferences": {
#         "communication_preferences": {
#             "telegram": {
#                 "preferred_encoding": [
#                     {
#                         "preference": "morse",
#                         "sentence_preference_revealed": "Morse, please.",
#                     }
#                 ],
#                 "favorite_telegram_operators": None,
#                 "preferred_telegram_paper": [
#                     {
#                         "preference": "Daredevil",
#                         "sentence_preference_revealed": 'Shall I use our "Daredevil" paper for this daring message?',
#                     }
#                 ],
#             },
#             "morse_code": {
#                 "preferred_key_type": [
#                     {
#                         "preference": "straight key",
#                         "sentence_preference_revealed": "I love using a straight key.",
#                     }
#                 ],
#                 "favorite_morse_abbreviations": None,
#             },
#             "semaphore": {"preferred_flag_color": None, "semaphore_skill_level": None},
#         },
#         "trust_fall_preferences": {
#             "preferred_fall_height": [
#                 {
#                     "preference": "higher",
#                     "sentence_preference_revealed": "I'm ready for a higher fall.",
#                 }
#             ],
#             "trust_level": None,
#             "preferred_catching_technique": [
#                 {
#                     "preference": "diamond formation",
#                     "sentence_preference_revealed": "I prefer the diamond formation for catching.",
#                 }
#             ],
#         },
#     }
# }