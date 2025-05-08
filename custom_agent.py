# File: custom_agent.py
from datetime import datetime
from typing import Tuple, Optional, Any
import re

from langchain_experimental.generative_agents.generative_agent import GenerativeAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils import BColors

class AutonomousGenerativeAgent(GenerativeAgent):
    def __init__(self, **data: Any):
        super().__init__(**data)
        print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Initialized AutonomousGenerativeAgent, "
              f"using base GenerativeAgent logic.{BColors.ENDC}")

    def get_interpreted_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str, bool]:
        call_time = now or datetime.now()
        print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling parent's generate_reaction for: '{observation[:50]}...'{BColors.ENDC}")

        is_dialogue_flag, result_str = super().generate_reaction(observation, now=call_time)

        estimated_observation_importance = False
        try:
            if self.llm:
                poignancy_prompt_str = (
                    "Is the following observation mundane (e.g., brushing teeth) or poignant (e.g., a major life event)? "
                    "Answer with 'mundane' or 'poignant'.\nObservation: {observation_text}\nAnswer:"
                )
                prompt = PromptTemplate.from_template(poignancy_prompt_str)
                chain = LLMChain(llm=self.llm, prompt=prompt)
                poignancy_result = chain.run(observation_text=observation).strip().lower()
                if "poignant" in poignancy_result:
                    estimated_observation_importance = True
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy estimated as '{poignancy_result}', API important flag: {estimated_observation_importance}{BColors.ENDC}")
        except Exception as e_poignancy:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Could not estimate observation poignancy for API flag: {e_poignancy}{BColors.ENDC}")

        content = self._clean_response(result_str)

        if is_dialogue_flag:
            actual_dialogue = content[len("said "):].strip() if content.lower().startswith("said ") else content
            print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Interpreted as SAY, Content: '{actual_dialogue}'{BColors.ENDC}")
            return "SAY", actual_dialogue, estimated_observation_importance
        else:
            if not content.strip() or content.lower() == "none":
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): Interpreted as IGNORE.{BColors.ENDC}")
                return "IGNORE", "", estimated_observation_importance
            else:
                print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Interpreted as DO (reaction), Content: '{content}'{BColors.ENDC}")
                return "DO", content, estimated_observation_importance
