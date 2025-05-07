# File: custom_agent.py
from datetime import datetime
from typing import Optional, List, Tuple, Any

from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document # Keep this import

# ANSI Color Codes (copy from main.py or redefine)
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    IMPORTANCE_HIGH = OKGREEN
    IMPORTANCE_MEDIUM = WARNING
    IMPORTANCE_LOW = FAIL
    METADATA_KEY = OKCYAN
    METADATA_VALUE = OKBLUE
    CONTENT_COLOR = ENDC
    SEPARATOR = DIM


class AutonomousGenerativeAgent(GenerativeAgent):
    """
    An autonomous agent that decides whether to SAY, THINK, DO, or IGNORE
    based on observation and its personality.
    """

    decision_chain: Optional[LLMChain] = None
    thought_chain: Optional[LLMChain] = None
    action_chain: Optional[LLMChain] = None
    status_update_chain: Optional[LLMChain] = None # Optional: for updating status after DO

    def _initialize_chains(self):
        """Initialize the specific chains needed for autonomous reactions."""
        if not self.decision_chain:
            # DECISION PROMPT
            decision_template = (
                "You are {agent_name}.\n"
                "Your core characteristics are: {agent_traits}\n"
                "It is {current_time}.\n"
                "Your status is: {agent_status}\n"
                "You observe: {observation}\n"
                "Relevant recent memories:\n{memory_context}\n\n"
                "Considering your personality (especially traits like {agent_traits}), the observation, and recent memories, "
                "what is the *most likely type* of immediate reaction you would have? Choose *one* from: "
                "'SAY' (speak aloud), 'THINK' (internal thought only), 'DO' (perform a physical action), 'IGNORE' (no significant reaction, remain in current status)."
                "\nReaction Type Choice:"
            )
            decision_prompt = PromptTemplate(
                input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context"],
                template=decision_template
            )
            self.decision_chain = LLMChain(llm=self.llm, prompt=decision_prompt, verbose=self.verbose)

        if not self.thought_chain:
            # THOUGHT PROMPT
            thought_template = (
                "You are {agent_name}.\n"
                "Your core characteristics are: {agent_traits}\n"
                "It is {current_time}.\n"
                "Your status is: {agent_status}\n"
                "You observe: {observation}\n"
                "Relevant recent memories:\n{memory_context}\n\n"
                "Considering your personality and the situation, what is your *internal thought* or *assessment* right now in response to the observation? "
                "Describe the thought concisely. Do *not* describe actions or speech. Example: (Internal thought) That seems suspicious."
                "\nInternal Thought:"
            )
            thought_prompt = PromptTemplate(
                input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context"],
                template=thought_template
            )
            self.thought_chain = LLMChain(llm=self.llm, prompt=thought_prompt, verbose=self.verbose)

        if not self.action_chain:
            # ACTION PROMPT
            action_template = (
                "You are {agent_name}.\n"
                "Your core characteristics are: {agent_traits}\n"
                "It is {current_time}.\n"
                "Your status is: {agent_status}\n"
                "You observe: {observation}\n"
                "Relevant recent memories:\n{memory_context}\n\n"
                "Considering your personality and the situation, what *physical action* do you take in immediate response to the observation? "
                "Describe the action concisely as if narrating it. Example: I shift my weight uneasily. / I draw my sword."
                "\nAction Taken:"
            )
            action_prompt = PromptTemplate(
                 input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context"],
                template=action_template
            )
            self.action_chain = LLMChain(llm=self.llm, prompt=action_prompt, verbose=self.verbose)

        # Optional: Chain to update status after an action
        if not self.status_update_chain:
             status_update_template = (
                 "You are {agent_name}.\n"
                 "Your core characteristics are: {agent_traits}\n"
                 "Your previous status was: {previous_status}\n"
                 "You just performed the action: {action_taken}\n"
                 "Based on this action, what is your concise, updated status? Describe it in the first person (e.g., 'Standing alert.', 'Sitting and observing.')."
                 "\nUpdated Status:"
             )
             status_update_prompt = PromptTemplate(
                 input_variables=["agent_name", "agent_traits", "previous_status", "action_taken"],
                 template=status_update_template
             )
             self.status_update_chain = LLMChain(llm=self.llm, prompt=status_update_prompt, verbose=self.verbose)


    def _fetch_context(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str]:
        """Helper to get memory context and current time string."""
        # Ensure retriever k value is handled if temporarily changed
        original_k = -1
        retriever = self.memory.memory_retriever
        if hasattr(retriever, 'k'):
            original_k = retriever.k # Store original K if needed
        # Add logic here if you want to temporarily change K for context fetching

        try:
            relevant_memories = self.fetch_memories(observation, now)
            memory_context = "\n".join(
                [f"- {m.page_content.strip()}" for m in relevant_memories]
            )
            if not memory_context.strip():
                memory_context = "No relevant memories."
        finally:
             if original_k != -1 and hasattr(retriever, 'k'):
                retriever.k = original_k # Restore original K

        current_time_str = (now or datetime.now()).strftime("%B %d, %Y, %I:%M:%S %p")
        return memory_context, current_time_str


    def _decide_reaction_type(self, observation: str, now: Optional[datetime] = None) -> str:
        """Uses an LLM call to determine the *type* of reaction."""
        self._initialize_chains() # Ensure chains are ready
        now = now or datetime.now()
        memory_context, current_time_str = self._fetch_context(observation, now)
        agent_summary = self.get_summary(now=now) # Get current summary

        result = self.decision_chain.run(
            agent_name=self.name,
            agent_traits=self.traits,
            current_time=current_time_str,
            agent_status=self.status,
            observation=observation,
            memory_context=memory_context,
            stop=["\n"]
        )
        decision = result.strip().upper()
        print(f"{BColors.OKCYAN}DEBUG (Agent {self.name}): Decided reaction type: '{decision}' for observation: '{observation[:50]}...'{BColors.ENDC}", flush=True)

        # Validation
        valid_types = ["SAY", "THINK", "DO", "IGNORE"]
        if decision not in valid_types:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): LLM decision '{decision}' invalid, defaulting to THINK.{BColors.ENDC}", flush=True)
            decision = "THINK" # Default to THINK instead of IGNORE might be safer
        return decision


    def generate_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Generates a reaction (SAY, THINK, DO, or IGNORE) based on the observation.
        Returns: (observation_was_important, reaction_output_string)
        """
        self._initialize_chains() # Ensure chains are ready
        call_time = now or datetime.now()
        reaction_type = self._decide_reaction_type(observation, call_time)

        # First, assess observation importance independently (like original method)
        # This uses the _compute_agent_summary logic internally
        # We need to call the poignancy chain directly or replicate its logic
        # For simplicity, let's call the base class's importance check first
        # Note: This might need access to the base _generate_reaction or similar logic if it's not public
        # Simplified: Let's run the poignancy check from memory directly if possible
        observation_poignancy = 0
        try:
             if hasattr(self.memory, 'score_memory_importance'):
                 observation_poignancy = self.memory.score_memory_importance(observation)
             else: # Fallback if method isn't available or named differently
                 # Simplified poignancy - less accurate than base method's chain
                 poignancy_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
                     "Rate the poignancy of this observation on a scale of 1 to 10 (integer): {observation}\nRating:"))
                 try:
                    observation_poignancy = int(poignancy_chain.run(observation=observation).strip())
                 except ValueError:
                    observation_poignancy = 3 # Default if parsing fails
             print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy rated: {observation_poignancy}{BColors.ENDC}", flush=True)
        except Exception as e:
             print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to score observation poignancy: {e}{BColors.ENDC}", flush=True)
             observation_poignancy = 3 # Default importance

        # Add observation memory IF it's deemed important enough OR led to a reaction
        add_observation_memory = observation_poignancy >= self.memory.reflection_threshold if self.memory.reflection_threshold else False

        # Now generate reaction based on decision
        reaction_output = ""

        if reaction_type == "IGNORE":
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Chose to IGNORE.{BColors.ENDC}", flush=True)
            reaction_output = ""
            # If ignoring, maybe don't add the observation memory unless very poignant?
            # Let's stick to adding based on poignancy for now.
            if add_observation_memory:
                 self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)

        else: # THINK, DO, or SAY involve a reaction, so observation is likely relevant
             add_observation_memory = True # Override: if reacting, observation is important
             memory_context, current_time_str = self._fetch_context(observation, call_time)
             agent_summary = self.get_summary(now=call_time) # Re-fetch summary if needed

             if reaction_type == "THINK":
                 thought_text = self.thought_chain.run(
                     agent_name=self.name,
                     agent_traits=self.traits,
                     current_time=current_time_str,
                     agent_status=self.status,
                     observation=observation,
                     memory_context=memory_context,
                 ).strip()
                 print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated thought: {thought_text}{BColors.ENDC}", flush=True)
                 # Add thought to memory
                 memory_to_add = f"(Internal thought) {thought_text}"
                 self.memory.add_memory(memory_to_add, now=call_time) # Importance will be scored by memory class
                 reaction_output = f"THINK: {thought_text}"

             elif reaction_type == "DO":
                 action_text = self.action_chain.run(
                      agent_name=self.name,
                      agent_traits=self.traits,
                      current_time=current_time_str,
                      agent_status=self.status,
                      observation=observation,
                      memory_context=memory_context,
                 ).strip()
                 print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Generated action: {action_text}{BColors.ENDC}", flush=True)
                 # Add action to memory
                 memory_to_add = f"(Action) {action_text}"
                 self.memory.add_memory(memory_to_add, now=call_time)
                 # Update status based on action
                 try:
                    updated_status = self.status_update_chain.run(
                        agent_name=self.name,
                        agent_traits=self.traits,
                        previous_status=self.status,
                        action_taken=action_text,
                        stop=["\n"]
                    ).strip()
                    if updated_status:
                        self.status = updated_status
                        print(f"{BColors.DIM}DEBUG (Agent {self.name}): Status updated to: '{self.status}'{BColors.ENDC}", flush=True)
                    else:
                         print(f"{BColors.WARNING}WARN (Agent {self.name}): Status update chain returned empty.{BColors.ENDC}", flush=True)
                         self.status = f"Just {action_text}" # Fallback status update

                 except Exception as e:
                    print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to update status via LLM: {e}. Using fallback.{BColors.ENDC}", flush=True)
                    self.status = f"Just {action_text}" # Fallback status update if chain fails

                 reaction_output = f"DO: {action_text}"

             elif reaction_type == "SAY":
                 # Replicate core dialogue generation logic from base class
                 # This requires the dialogue prompt and chain execution
                 # We need to access or replicate _get_dialogue_prompt and the main dialogue chain
                 try:
                    # Conceptual replication - assumes dialogue_chain exists and is set up like base class
                    if not hasattr(self, 'chain'): # Check if base chain exists
                         raise AttributeError("Dialogue chain 'chain' not found on agent.")

                    dialogue_result = self.chain.run( # Use the main dialogue chain
                        agent_summary=agent_summary,
                        current_time=current_time_str,
                        memory_context=memory_context,
                        agent_status=self.status, # Use current status
                        observation=observation,
                        stop=["\n"] # Often helpful
                        ).strip()

                    # Base class parsing logic (simplified)
                    if '"' in dialogue_result:
                        dialogue_text = dialogue_result.split('"')[-2]
                    else:
                        dialogue_text = dialogue_result

                    print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Generated dialogue: {dialogue_text}{BColors.ENDC}", flush=True)
                    # Add dialogue to memory
                    memory_to_add = f'{self.name} said "{dialogue_text}"'
                    self.memory.add_memory(memory_to_add, now=call_time)
                    reaction_output = f"SAY: {dialogue_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Failed during dialogue generation step: {e}. Defaulting to THINK.{BColors.ENDC}", flush=True)
                     # Fallback to THINK if SAY fails
                     reaction_output = "THINK: (Failed to formulate a verbal response)"
                     self.memory.add_memory("(Internal thought) Failed to formulate a verbal response.", now=call_time)

             # Add the original observation to memory because a reaction occurred
             if add_observation_memory:
                  self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)


        # Return value: was observation important enough on its own merit? + reaction string
        return (observation_poignancy > 3, reaction_output) # Example threshold for importance return

    # Override generate_dialogue_response to use the new logic
    def generate_dialogue_response(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Overrides base method to use generate_reaction and parse output for dialogue."""
        observation_important, reaction_string = self.generate_reaction(observation, now)

        # Only return the dialogue part if the reaction was SAY
        if reaction_string.startswith("SAY:"):
            dialogue = reaction_string[len("SAY:"):].strip()
            return observation_important, dialogue
        else:
            # If not SAY, return empty string for dialogue, but keep importance flag
            return observation_important, ""
