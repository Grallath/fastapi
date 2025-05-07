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
    cached_summary: Optional[str] = None  # Cache for the summary to avoid duplicate generation
    cached_summary_time: Optional[datetime] = None  # When the summary was last generated
    cached_relationship_context: Optional[str] = None  # Cache for relationship context
    cached_relationship_time: Optional[datetime] = None  # When the relationship context was last generated

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
                "Relevant recent memories:\n{memory_context}\n"
                "Relationship context: {relationship_context}\n\n"
                "Considering your personality and the situation, what is your *internal thought* or *assessment* right now in response to the observation? "
                "Describe the thought concisely. Do *not* describe actions or speech. Example: (Internal thought) That seems suspicious."
                "\nInternal Thought:"
            )
            thought_prompt = PromptTemplate(
                input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context", "relationship_context"],
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
                "Relevant recent memories:\n{memory_context}\n"
                "Relationship context: {relationship_context}\n\n"
                "Considering your personality and the situation, what *physical action* do you take in immediate response to the observation? "
                "Describe the action concisely as if narrating it. Example: I shift my weight uneasily. / I draw my sword."
                "\nAction Taken:"
            )
            action_prompt = PromptTemplate(
                 input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context", "relationship_context"],
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
             print(f"{BColors.DIM}DEBUG (Agent {self.name}): Status update chain initialized.{BColors.ENDC}")

    def _get_entity_from_observation(self, observation: str) -> str:
        """Extract the main entity from an observation."""
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        try:
            entity = self.chain(prompt).run(observation=observation).strip()
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Identified entity: '{entity}' from observation{BColors.ENDC}", flush=True)
            return entity
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to extract entity: {e}. Using fallback.{BColors.ENDC}", flush=True)
            return "unknown entity"

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        """Determine what the entity is doing in the observation."""
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        try:
            action = self.chain(prompt).run(entity=entity_name, observation=observation).strip()
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Identified action: '{entity_name} is {action}'{BColors.ENDC}", flush=True)
            return action
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to extract action: {e}. Using fallback.{BColors.ENDC}", flush=True)
            return "present"

    def summarize_related_memories(self, observation: str, now: Optional[datetime] = None) -> str:
        """Summarize memories that are most relevant to an observation, focusing on relationships."""
        # Check if we have a recent cached relationship context
        current_time = now or datetime.now()
        if (self.cached_relationship_context is not None and 
            self.cached_relationship_time is not None and
            (current_time - self.cached_relationship_time).total_seconds() < 1.0):
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Using cached relationship context{BColors.ENDC}", flush=True)
            return self.cached_relationship_context
            
        prompt = PromptTemplate.from_template(
            """
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
"""
        )
        
        try:
            entity_name = self._get_entity_from_observation(observation)
            entity_action = self._get_entity_action(observation, entity_name)
            q1 = f"What is the relationship between {self.name} and {entity_name}"
            q2 = f"{entity_name} is {entity_action}"
            
            # Get relationship context
            relationship_context = self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()
            
            if not relationship_context or relationship_context.lower() == "none" or relationship_context.lower() == "no relevant context":
                relationship_context = f"No known relationship with {entity_name}."
                
            print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated relationship context: {relationship_context}{BColors.ENDC}", flush=True)
            
            # Cache the result
            self.cached_relationship_context = relationship_context
            self.cached_relationship_time = current_time
            
            return relationship_context
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to summarize related memories: {e}. Using fallback.{BColors.ENDC}", flush=True)
            return "No specific relationship information available."

    def _fetch_context(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str]:
        """Helper to get memory context and current time string."""
        if not self.memory:
             raise ValueError("Agent memory is not initialized.")

        original_k = -1
        retriever = self.memory.memory_retriever
        if hasattr(retriever, 'k'):
            original_k = retriever.k

        try:
            # Call fetch_memories on the memory object
            relevant_memories: List[Document] = self.memory.fetch_memories(observation, now)

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

    # Override get_summary to use our cached version during a single reaction generation
    def get_summary(self, force_refresh: bool = False, now: Optional[datetime] = None) -> str:
        """Return a descriptive summary of the agent, using cache when appropriate."""
        current_time = now or datetime.now()
        
        # If we have a cached summary from the same reaction cycle, use it
        if (not force_refresh and 
            self.cached_summary is not None and 
            self.cached_summary_time is not None and
            (current_time - self.cached_summary_time).total_seconds() < 1.0):  # Use cache if less than 1 second old
            return self.cached_summary
            
        # Otherwise, get a fresh summary from the parent class
        summary = super().get_summary(force_refresh=force_refresh, now=current_time)
        
        # Cache the result
        self.cached_summary = summary
        self.cached_summary_time = current_time
        
        return summary

    def _decide_reaction_type(self, observation: str, now: Optional[datetime] = None) -> str:
        """Uses an LLM call to determine the *type* of reaction."""
        self._initialize_chains() # Ensure chains are ready
        now = now or datetime.now()
        memory_context, current_time_str = self._fetch_context(observation, now)
        agent_summary = self.get_summary(now=now, force_refresh=False)

        try:
            result = self.decision_chain.run(
                agent_name=self.name,
                agent_traits=self.traits,
                current_time=current_time_str,
                agent_status=self.status,
                observation=observation,
                memory_context=memory_context,
            )
            decision = result.strip().split('\n')[0].replace('"', '').replace("'", '').strip().upper()
        except Exception as e:
            print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during decision chain: {e}{BColors.ENDC}", flush=True)
            decision = "THINK"

        print(f"{BColors.OKCYAN}DEBUG (Agent {self.name}): Decided reaction type: '{decision}' for observation: '{observation[:50]}...'{BColors.ENDC}", flush=True)

        # Validation
        valid_types = ["SAY", "THINK", "DO", "IGNORE"]
        if decision not in valid_types:
            found = False
            for vt in valid_types:
                if vt in result.strip().upper():
                    decision = vt
                    found = True
                    print(f"{BColors.DIM}DEBUG (Agent {self.name}): Corrected decision to '{decision}' from raw result '{result.strip()}'{BColors.ENDC}")
                    break
            if not found:
                print(f"{BColors.WARNING}WARN (Agent {self.name}): LLM decision '{result}' invalid, defaulting to THINK.{BColors.ENDC}", flush=True)
                decision = "THINK"
        return decision


    def generate_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Generates a reaction (SAY, THINK, DO, or IGNORE) based on the observation.
        Returns: (observation_was_important, reaction_output_string)
        """
        if not self.memory:
             raise ValueError("Agent memory is not initialized before generating reaction.")

        # Reset the caches at the start of a new reaction generation
        self.cached_summary = None
        self.cached_summary_time = None
        self.cached_relationship_context = None
        self.cached_relationship_time = None

        self._initialize_chains()
        call_time = now or datetime.now()
        reaction_type = self._decide_reaction_type(observation, call_time)

        # Get relationship context for the observation
        relationship_context = self.summarize_related_memories(observation, call_time)

        # Assess observation importance
        observation_poignancy = 0
        add_observation_memory = False
        reflection_enabled = self.memory.reflection_threshold is not None and self.memory.reflection_threshold > 0

        if reflection_enabled:
            try:
                 if hasattr(self.memory, 'score_memory_importance'):
                     observation_poignancy = self.memory.score_memory_importance(observation)
                 else:
                    print(f"{BColors.WARNING}WARN (Agent {self.name}): memory.score_memory_importance not found. Using fallback scoring.{BColors.ENDC}")
                    poignancy_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
                        "Rate the poignancy of this observation on a scale of 1 to 10 (integer): {observation}\nRating:"))
                    try:
                       poignancy_result = poignancy_chain.run(observation=observation).strip()
                       observation_poignancy = int(poignancy_result)
                    except ValueError:
                       print(f"{BColors.WARNING}WARN (Agent {self.name}): Fallback poignancy scoring failed to parse integer from '{poignancy_result}'. Defaulting to 3.{BColors.ENDC}")
                       observation_poignancy = 3
                 # Compare with the threshold from the memory object
                 add_observation_memory = observation_poignancy >= self.memory.reflection_threshold
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy rated: {observation_poignancy} (Threshold: {self.memory.reflection_threshold}, AddMem: {add_observation_memory}){BColors.ENDC}", flush=True)
            except Exception as e:
                 print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to score observation poignancy: {e}. Defaulting importance based on reaction.{BColors.ENDC}", flush=True)
                 observation_poignancy = 3
                 add_observation_memory = reaction_type != "IGNORE"
        else:
            add_observation_memory = reaction_type != "IGNORE"
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Reflection disabled. AddMem based on reaction: {add_observation_memory}{BColors.ENDC}")

        # Generate reaction based on decision
        reaction_output = ""

        if reaction_type == "IGNORE":
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Chose to IGNORE.{BColors.ENDC}", flush=True)
            reaction_output = ""
            # Add observation memory ONLY if it independently crossed the threshold
            if add_observation_memory and reflection_enabled:
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Adding poignant observation to memory despite IGNORE.{BColors.ENDC}", flush=True)
                 # --- CORRECTED CALL ---
                 self.memory.add_memory(observation, now=call_time) # REMOVED importance kwarg
                 # --- END CORRECTION ---
            return (add_observation_memory, reaction_output)

        else: # THINK, DO, or SAY involve a reaction
             memory_context, current_time_str = self._fetch_context(observation, call_time)
             # Use the cached summary from the decision step
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
                         relationship_context=relationship_context,
                     ).strip()
                     print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated thought: {thought_text}{BColors.ENDC}", flush=True)
                     memory_to_add = f"(Internal thought) {thought_text}"
                     # Add thought memory (no importance needed here)
                     self.memory.add_memory(memory_to_add, now=call_time)
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
                          relationship_context=relationship_context,
                     ).strip()
                     print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Generated action: {action_text}{BColors.ENDC}", flush=True)
                     memory_to_add = f"(Action) {action_text}"
                     # Add action memory
                     self.memory.add_memory(memory_to_add, now=call_time)
                     # Update status based on action
                     try:
                        updated_status = self.status_update_chain.run(
                            agent_name=self.name,
                            agent_traits=self.traits,
                            previous_status=self.status,
                            action_taken=action_text,
                        ).strip().split('\n')[0].replace('"', '').replace("'", '').strip()
                        if updated_status:
                            print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Status updated from '{self.status}' to '{updated_status}'{BColors.ENDC}", flush=True)
                            self.status = updated_status
                     except Exception as e_status:
                        print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to update status via LLM: {e_status}. Using fallback.{BColors.ENDC}", flush=True)
                        self.status = f"Just {action_text}"

                     reaction_output = f"DO: {action_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during action chain: {e}{BColors.ENDC}", flush=True)
                     reaction_output = "DO: (Error generating action)"
                     self.memory.add_memory("(Action) Error generating action.", now=call_time)

             elif reaction_type == "SAY":
                 try:
                    # Use the base class's generate_dialogue_response
                    print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling super().generate_dialogue_response for SAY.{BColors.ENDC}", flush=True)
                    dialogue_important_flag, dialogue_text = super().generate_dialogue_response(observation, call_time)
                    # Base class handles adding observation and response to memory here
                    print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Generated dialogue: {dialogue_text}{BColors.ENDC}", flush=True)
                    reaction_output = f"SAY: {dialogue_text}"

                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Failed during SAY generation step (super call): {e}. Defaulting to THINK.{BColors.ENDC}", flush=True)
                     reaction_output = "THINK: (Failed to formulate a verbal response)"
                     self.memory.add_memory("(Internal thought) Failed to formulate a verbal response.", now=call_time)

             # Add the original observation memory if needed (based on earlier check or reaction)
             if add_observation_memory:
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Adding observation memory (Poignancy: {observation_poignancy}, Reaction: {reaction_type}).{BColors.ENDC}", flush=True)
                 # --- CORRECTED CALL ---
                 self.memory.add_memory(observation, now=call_time) # REMOVED importance kwarg
                 # --- END CORRECTION ---

        # Final importance flag: based on initial observation scoring
        final_importance_flag = add_observation_memory
        return (final_importance_flag, reaction_output)


    def generate_dialogue_response(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Overrides base method to use generate_reaction BUT only returns the spoken part
        if the reaction type was 'SAY'. Otherwise returns empty dialogue.
        The boolean indicates if the original observation was deemed important based on its poignancy score.
        """
        observation_important, reaction_string = self.generate_reaction(observation, now)

        if reaction_string.startswith("SAY:"):
            dialogue = reaction_string[len("SAY:"):].strip()
            return observation_important, dialogue
        else:
            return observation_important, ""
