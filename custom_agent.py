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

    def get_interpreted_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str, bool, int]:
        call_time = now or datetime.now()
        print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling parent's generate_reaction for: '{observation[:50]}...'{BColors.ENDC}")

        is_dialogue_flag, result_str = super().generate_reaction(observation, now=call_time)

        estimated_observation_importance = False
        poignancy_rating = 0
        try:
            if self.llm:
                # Use the 1-10 scale for poignancy rating
                poignancy_prompt_str = (
                    "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant "
                    "(e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. "
                    "Respond with a single integer.\n\nMemory: {observation_text}\n\nRating: "
                )
                prompt = PromptTemplate.from_template(poignancy_prompt_str)
                chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
                # Print the prompt being sent to the LLM
                print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Sending poignancy rating prompt to LLM:{BColors.ENDC}")
                print(f"{BColors.HEADER}{poignancy_prompt_str.format(observation_text=observation[:100] + '...' if len(observation) > 100 else observation)}{BColors.ENDC}")

                # Run the chain and get the result
                print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Running LLMChain for poignancy rating...{BColors.ENDC}")
                # Use the __call__ method instead of run to get more detailed logging
                chain_response = chain({"observation_text": observation})

                # Print the raw chain response to debug
                print(f"\nDEBUG: Raw chain response: {chain_response}\n")

                # Get the text from the response
                poignancy_result = chain_response.get('text', '').strip()

                # Print the raw result immediately
                print(f"\n!!! RAW POIGNANCY RESULT: '{poignancy_result}' !!!\n")

                # Print the raw response from the LLM with special formatting to make it stand out
                print(f"{BColors.OKBLUE}\n====================\nDEBUG (Agent {self.name}): Raw LLM response for poignancy rating:\n{BColors.BOLD}{poignancy_result}{BColors.ENDC}{BColors.OKBLUE}\n===================={BColors.ENDC}")

                # Also log to standard output to ensure it's captured
                import sys
                sys.stdout.write(f"\nRAW LLM RATING RESPONSE: {poignancy_result}\n")
                sys.stdout.flush()

                # Extract the numeric rating
                try:
                    # Try to extract just the number from the response
                    import re
                    print(f"\nDEBUG: Attempting to extract number from: '{poignancy_result}'\n")

                    # Try different regex patterns
                    # First try to match just a number from 1-10
                    number_match = re.search(r'\b([1-9]|10)\b', poignancy_result)
                    print(f"\nDEBUG: First regex match result: {number_match}\n")

                    # If that doesn't work, try a more general pattern
                    if not number_match:
                        number_match = re.search(r'(\d+)', poignancy_result)
                        print(f"\nDEBUG: Second regex match result: {number_match}\n")

                    if number_match:
                        poignancy_rating = int(number_match.group(1))
                        print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Successfully extracted rating: {poignancy_rating} from regex match{BColors.ENDC}")
                        print(f"\n*** EXTRACTED RATING: {poignancy_rating} ***\n")
                    else:
                        try:
                            print(f"\nDEBUG: Attempting to convert full response to int: '{poignancy_result}'\n")
                            poignancy_rating = int(poignancy_result)
                            print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Successfully converted full response to rating: {poignancy_rating}{BColors.ENDC}")
                            print(f"\n*** CONVERTED RATING: {poignancy_rating} ***\n")
                        except ValueError as ve:
                            print(f"{BColors.WARNING}DEBUG (Agent {self.name}): Could not parse rating from: '{poignancy_result}', defaulting to 0. Error: {ve}{BColors.ENDC}")
                            print(f"\n!!! FAILED TO PARSE RATING: '{poignancy_result}' !!!\n")
                            poignancy_rating = 0
                except (ValueError, TypeError) as e:
                    # If we can't parse a number, default to 0
                    print(f"{BColors.WARNING}DEBUG (Agent {self.name}): Error parsing rating: {e}, defaulting to 0{BColors.ENDC}")
                    print(f"\n!!! EXCEPTION PARSING RATING: {e} !!!\n")
                    poignancy_rating = 0

                # Consider ratings of 6 or higher as important
                if poignancy_rating >= 6:
                    estimated_observation_importance = True

                print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Final observation poignancy rating: {poignancy_rating}/10, API important flag: {estimated_observation_importance}{BColors.ENDC}")
        except Exception as e_poignancy:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Could not estimate observation poignancy for API flag: {e_poignancy}{BColors.ENDC}")

        content = self._clean_response(result_str)

        if is_dialogue_flag:
            actual_dialogue = content[len("said "):].strip() if content.lower().startswith("said ") else content
            print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Interpreted as SAY, Content: '{actual_dialogue}'{BColors.ENDC}")
            return "SAY", actual_dialogue, estimated_observation_importance, poignancy_rating
        else:
            if not content.strip() or content.lower() == "none":
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): Interpreted as IGNORE.{BColors.ENDC}")
                return "IGNORE", "", estimated_observation_importance, poignancy_rating
            else:
                print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Interpreted as DO (reaction), Content: '{content}'{BColors.ENDC}")
                return "DO", content, estimated_observation_importance, poignancy_rating
