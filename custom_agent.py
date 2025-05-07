# File: custom_agent.py
from datetime import datetime
from typing import Optional, List, Tuple, Any

# Note: GenerativeAgent and GenerativeAgentMemory are experimental
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain # Ensure LLMChain is imported
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document

# ANSI Color Codes
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
        if not self.llm:
             raise ValueError("Agent LLM is not initialized.")

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
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Decision chain initialized.{BColors.ENDC}")


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
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Thought chain initialized.{BColors.ENDC}")


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
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Action chain initialized.{BColors.ENDC}")


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
             print(f"{BColors.DIM}DEBUG (Agent {self.name}): Status Update chain initialized.{BColors.ENDC}")


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
        # Use the base class method to get the summary respecting refresh logic
        agent_summary = self.get_summary(now=now, force_refresh=False)

        try:
            result = self.decision_chain.run(
                agent_name=self.name,
                agent_traits=self.traits,
                current_time=current_time_str,
                agent_status=self.status,
                observation=observation,
                memory_context=memory_context,
                # stop=["\n"] # Keep stop token
            )
            decision = result.strip().upper().split('\n')[0].replace('"', '').replace("'", '') # Get first line, clean up
        except Exception as e:
            print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during decision chain: {e}{BColors.ENDC}", flush=True)
            decision = "THINK" # Default if chain fails

        print(f"{BColors.OKCYAN}DEBUG (Agent {self.name}): Decided reaction type: '{decision}' for observation: '{observation[:50]}...'{BColors.ENDC}", flush=True)

        # Validation
        valid_types = ["SAY", "THINK", "DO", "IGNORE"]
        if decision not in valid_types:
            # Try to find a valid type within the response if simple strip failed
            found = False
            for vt in valid_types:
                if vt in decision:
                    decision = vt
                    found = True
                    print(f"{BColors.DIM}DEBUG (Agent {self.name}): Corrected decision to '{decision}'{BColors.ENDC}")
                    break
            if not found:
                print(f"{BColors.WARNING}WARN (Agent {self.name}): LLM decision '{result}' invalid, defaulting to THINK.{BColors.ENDC}", flush=True)
                decision = "THINK" # Default to THINK if still invalid
        return decision


    def generate_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Generates a reaction (SAY, THINK, DO, or IGNORE) based on the observation.
        Returns: (observation_was_important, reaction_output_string)
        """
        self._initialize_chains() # Ensure chains are ready
        call_time = now or datetime.now()
        reaction_type = self._decide_reaction_type(observation, call_time)

        # Assess observation importance
        observation_poignancy = 0
        add_observation_memory = False
        if self.memory.reflection_threshold is not None: # Only score if reflection is enabled
            try:
                 if hasattr(self.memory, 'score_memory_importance'):
                     observation_poignancy = self.memory.score_memory_importance(observation)
                 else: # Fallback if method isn't available
                    # This fallback is less accurate
                    poignancy_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
                        "Rate the poignancy of this observation on a scale of 1 to 10 (integer): {observation}\nRating:"))
                    try:
                       observation_poignancy = int(poignancy_chain.run(observation=observation).strip())
                    except ValueError:
                       observation_poignancy = 3
                 add_observation_memory = observation_poignancy >= self.memory.reflection_threshold
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy rated: {observation_poignancy} (Threshold: {self.memory.reflection_threshold}){BColors.ENDC}", flush=True)
            except Exception as e:
                 print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to score observation poignancy: {e}. Defaulting importance.{BColors.ENDC}", flush=True)
                 observation_poignancy = 3 # Default importance score
                 add_observation_memory = reaction_type != "IGNORE" # Add if reacting, even if scoring failed

        # Now generate reaction based on decision
        reaction_output = ""

        if reaction_type == "IGNORE":
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Chose to IGNORE.{BColors.ENDC}", flush=True)
            reaction_output = ""
            if add_observation_memory:
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Adding poignant observation to memory despite IGNORE.{BColors.ENDC}", flush=True)
                 self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)
            # Observation itself wasn't reacted to, so return False for importance in this context?
            # Let's return True only if the observation *independently* crossed the threshold
            return (add_observation_memory, reaction_output)

        else: # THINK, DO, or SAY
             memory_context, current_time_str = self._fetch_context(observation, call_time)
             agent_summary = self.get_summary(now=call_time, force_refresh=False)

             if reaction_type == "THINK":
                 try:
                     thought_text = self.thought_chain.run(
                         agent_name=self.name,
                         agent_traits=self.traits,
                         current_time=current_time_str,
                         agent_status=self.status,
                         observation=observation,
                         memory_context=memory_context,
                     ).strip()
                     print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated thought: {thought_text}{BColors.ENDC}", flush=True)
                     memory_to_add = f"(Internal thought) {thought_text}"
                     self.memory.add_memory(memory_to_add, now=call_time) # Let memory score it
                     reaction_output = f"THINK: {thought_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during thought chain: {e}{BColors.ENDC}", flush=True)
                     reaction_output = "THINK: (Error generating thought)"
                     self.memory.add_memory("(Internal thought) Error generating thought.", now=call_time)


             elif reaction_type == "DO":
                 try:
                     action_text = self.action_chain.run(
                          agent_name=self.name,
                          agent_traits=self.traits,
                          current_time=current_time_str,
                          agent_status=self.status,
                          observation=observation,
                          memory_context=memory_context,
                     ).strip()
                     print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Generated action: {action_text}{BColors.ENDC}", flush=True)
                     memory_to_add = f"(Action) {action_text}"
                     self.memory.add_memory(memory_to_add, now=call_time)
                     # Update status
                     try:
                        updated_status = self.status_update_chain.run(
                            agent_name=self.name,
                            agent_traits=self.traits,
                            previous_status=self.status,
                            action_taken=action_text,
                            # stop=["\n"]
                        ).strip().split('\n')[0].replace('"', '').replace("'", '') # Clean up status
                        if updated_status:
                            self.status = updated_status
                            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Status updated to: '{self.status}'{BColors.ENDC}", flush=True)
                        else:
                             print(f"{BColors.WARNING}WARN (Agent {self.name}): Status update chain returned empty. Using fallback.{BColors.ENDC}", flush=True)
                             self.status = f"Just {action_text}" # Simple fallback

                     except Exception as e_status:
                        print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to update status via LLM: {e_status}. Using fallback.{BColors.ENDC}", flush=True)
                        self.status = f"Just {action_text}" # Fallback status update

                     reaction_output = f"DO: {action_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during action chain: {e}{BColors.ENDC}", flush=True)
                     reaction_output = "DO: (Error generating action)"
                     self.memory.add_memory("(Action) Error generating action.", now=call_time)


             elif reaction_type == "SAY":
                 try:
                    # Utilize the base class's dialogue generation mechanism
                    # This assumes _generate_reaction handles the core logic including memory adding
                    # We might need to call a lower-level method if _generate_reaction isn't suitable
                    # Let's try calling the internal _generate_reaction method if accessible
                    # WARNING: Accessing protected methods (_*) is generally discouraged as they might change.
                    # If this fails, we have to replicate the logic.
                    if hasattr(self, '_generate_reaction'):
                        print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling internal _generate_reaction for SAY.{BColors.ENDC}", flush=True)
                        # _generate_reaction expects specific args and returns (importance, reaction)
                        # We already have context, let's pass it
                        # This internal method likely handles adding observation and response to memory.
                        dialogue_important, dialogue_text = self._generate_reaction(
                            observation, call_time, memory_context # Pass context needed by base method
                        )
                        print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Generated dialogue: {dialogue_text}{BColors.ENDC}", flush=True)
                        reaction_output = f"SAY: {dialogue_text}"
                        # Ensure observation gets added if _generate_reaction doesn't
                        # (It should, but as a fallback check)
                        if not any(m.page_content == observation for m in self.memory.memory_retriever.vectorstore.docstore.search(observation)):
                             print(f"{BColors.WARNING}WARN (Agent {self.name}): Observation memory might not have been added by _generate_reaction. Adding manually.{BColors.ENDC}")
                             self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)

                    else:
                        # Fallback: Replicate simplified dialogue logic if _generate_reaction is not callable
                        print(f"{BColors.WARNING}WARN (Agent {self.name}): _generate_reaction not accessible. Using fallback dialogue logic.{BColors.ENDC}")
                        # This requires the dialogue prompt template from the base class or a similar one
                        dialogue_prompt = PromptTemplate.from_template( # Example prompt structure
                            "Summary of {agent_name}'s life: {agent_summary}\n"
                            "Current time: {current_time}\n{agent_name}'s status: {agent_status}\n"
                            "Relevant Memories: {memory_context}\n"
                            "Observation: {observation}\n\n"
                            "What would {agent_name} say? Output dialogue wrapped in quotes."
                        )
                        dialogue_chain = LLMChain(llm=self.llm, prompt=dialogue_prompt, verbose=self.verbose)
                        dialogue_result = dialogue_chain.run(
                             agent_name = self.name,
                             agent_summary=agent_summary,
                             current_time=current_time_str,
                             memory_context=memory_context,
                             agent_status=self.status,
                             observation=observation
                        ).strip()
                        # Simple parsing
                        if '"' in dialogue_result:
                            dialogue_text = dialogue_result.split('"')[-2]
                        else:
                            dialogue_text = dialogue_result
                        print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Generated dialogue (fallback): {dialogue_text}{BColors.ENDC}", flush=True)
                        # Manually add observation and response to memory
                        self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)
                        memory_to_add = f'{self.name} said "{dialogue_text}"'
                        self.memory.add_memory(memory_to_add, now=call_time)
                        reaction_output = f"SAY: {dialogue_text}"

                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Failed during SAY generation step: {e}. Defaulting to THINK.{BColors.ENDC}", flush=True)
                     reaction_output = "THINK: (Failed to formulate a verbal response)"
                     self.memory.add_memory("(Internal thought) Failed to formulate a verbal response.", now=call_time)
                     # Add observation if important
                     if add_observation_memory:
                         self.memory.add_memory(observation, now=call_time, importance=observation_poignancy)


        # Determine overall importance: based on initial observation scoring OR if any reaction occurred
        final_importance_flag = add_observation_memory or (reaction_type != "IGNORE")
        return (final_importance_flag, reaction_output)


    # Override generate_dialogue_response to use the new logic but maintain expected output
    def generate_dialogue_response(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Overrides base method to use generate_reaction BUT only returns the spoken part
        if the reaction type was 'SAY'. Otherwise returns empty dialogue.
        The boolean indicates if the original observation was deemed important enough to add to memory.
        """
        observation_important, reaction_string = self.generate_reaction(observation, now)

        if reaction_string.startswith("SAY:"):
            dialogue = reaction_string[len("SAY:"):].strip()
            # Ensure the observation memory was added if we are saying something
            # The generate_reaction method should handle this, but double-check needed if complex
            return observation_important, dialogue
        else:
            # If the agent decided to THINK, DO, or IGNORE, return no dialogue.
            # The boolean still reflects the independent importance of the observation.
            return observation_important, ""
