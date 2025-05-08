# File: custom_agent.py
from datetime import datetime
from typing import Tuple, Optional, Any
import re

# Directly inherit from the original Langchain experimental agent
from langchain_experimental.generative_agents.generative_agent import GenerativeAgent
# GenerativeAgentMemory is used by GenerativeAgent, so it's implicitly needed.

from utils import BColors # Keep BColors for logging

class AutonomousGenerativeAgent(GenerativeAgent):
    """
    This class now acts as a direct extension of the original
    langchain_experimental.generative_agents.GenerativeAgent.
    It uses the parent's reaction logic. The interpretation of its output
    (into SAY, THINK, DO, IGNORE) is handled at the API layer.
    """

    def __init__(self, **data: Any):
        super().__init__(**data)
        print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Initialized as AutonomousGenerativeAgent, "
              f"using base GenerativeAgent logic.{BColors.ENDC}")

    # All core methods like _generate_reaction, generate_reaction,
    # generate_dialogue_response, summarize_related_memories, get_summary, etc.,
    # are inherited directly from the parent `GenerativeAgent` class.

    # No custom chains, no custom decision logic.

    def get_interpreted_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str, bool]:
        """
        Calls the parent's `generate_reaction` and interprets its output for the API.
        The parent `generate_reaction` method handles memory saving.

        Returns:
            - reaction_type (str): Interpreted as "SAY", "DO", "IGNORE".
            - content (str): The dialogue or action description.
            - observation_was_important (bool): A flag indicating if the observation was likely important.
                                                This is an estimation, as the actual importance scoring
                                                for memory happens within GenerativeAgentMemory.
        """
        call_time = now or datetime.now()
        print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling parent's generate_reaction for: '{observation[:50]}...'{BColors.ENDC}")

        # is_dialogue_flag: bool, result_str: str
        # result_str is like "AgentName said ..." or "AgentName action_description"
        is_dialogue_flag, result_str = super().generate_reaction(observation, now=call_time)

        # Determine initial observation importance for API response (heuristic)
        # The actual importance for memory is handled by GenerativeAgentMemory.add_memory
        # This is a simplified check for the API response.
        estimated_observation_importance = False
        try:
            # Quick check using a simplified poignancy prompt if llm is available
            if self.llm:
                from langchain.prompts import PromptTemplate
                from langchain.chains import LLMChain

                # Simplified prompt for a quick check
                poignancy_prompt_str = (
                    "Is the following observation mundane (e.g., brushing teeth) or poignant (e.g., a major life event)? "
                    "Answer with 'mundane' or 'poignant'.\nObservation: {observation_text}\nAnswer:"
                )
                prompt = PromptTemplate.from_template(poignancy_prompt_str)
                chain = LLMChain(llm=self.llm, prompt=prompt) # Temporary chain
                poignancy_result = chain.run(observation_text=observation).strip().lower()
                if "poignant" in poignancy_result:
                    estimated_observation_importance = True
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy estimated as '{poignancy_result}', API important flag: {estimated_observation_importance}{BColors.ENDC}")
        except Exception as e_poignancy:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Could not estimate observation poignancy for API flag: {e_poignancy}{BColors.ENDC}")


        # Clean the response to remove the agent's name prefix if present
        # The parent's _clean_response method is:
        # re.sub(f"^{self.name} ", "", text.strip()).strip()
        content = self._clean_response(result_str)

        if is_dialogue_flag:
            # Parent's generate_reaction returns: True, f"{self.name} said {said_value}"
            # So `content` here would be `said {said_value}`.
            # We need to extract the actual said_value.
            if content.lower().startswith("said "):
                actual_dialogue = content[len("said "):].strip()
            else:
                actual_dialogue = content # Fallback if "said " prefix isn't there
            print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Interpreted as SAY, Content: '{actual_dialogue}'{BColors.ENDC}")
            return "SAY", actual_dialogue, estimated_observation_importance
        else:
            # Parent's generate_reaction returns: False, f"{self.name} {reaction}" or False, result (if no "REACT:" prefix)
            # `content` here is the reaction description.
            if not content.strip() or content.lower() == "none": # If LLM explicitly says "None" or empty after cleaning
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): Interpreted as IGNORE (empty or 'None' content).{BColors.ENDC}")
                return "IGNORE", "", estimated_observation_importance # No specific "THINK" type from original
            else:
                # This is a physical action or a more general reaction.
                # The original doesn't distinguish "DO" vs "THINK". We'll map "REACT" to "DO".
                print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Interpreted as DO (reaction), Content: '{content}'{BColors.ENDC}")
                return "DO", content, estimated_observation_importance
