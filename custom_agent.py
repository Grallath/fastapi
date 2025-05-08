# File: custom_agent.py
from datetime import datetime
from typing import Optional, List, Tuple, Any

from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document

# Import prompts and colors from new locations
from prompts import (
    DECISION_TEMPLATE, THOUGHT_TEMPLATE, ACTION_TEMPLATE, STATUS_UPDATE_TEMPLATE,
    ENTITY_EXTRACTION_TEMPLATE, ENTITY_ACTION_TEMPLATE, RELATIONSHIP_SUMMARY_TEMPLATE,
    POIGNANCY_SCORING_FALLBACK_TEMPLATE
)
from utils import BColors


class AutonomousGenerativeAgent(GenerativeAgent):
    decision_chain: Optional[LLMChain] = None
    thought_chain: Optional[LLMChain] = None
    action_chain: Optional[LLMChain] = None
    status_update_chain: Optional[LLMChain] = None
    cached_summary: Optional[str] = None
    cached_summary_time: Optional[datetime] = None
    cached_relationship_context: Optional[str] = None
    cached_relationship_time: Optional[datetime] = None

    def _initialize_chains(self):
        if not self.llm:
             raise ValueError("Agent LLM is not initialized.")

        if not self.decision_chain:
            decision_prompt = PromptTemplate(
                input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context"],
                template=DECISION_TEMPLATE
            )
            self.decision_chain = LLMChain(llm=self.llm, prompt=decision_prompt, verbose=self.verbose)
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Decision chain initialized.{BColors.ENDC}")

        if not self.thought_chain:
            thought_prompt = PromptTemplate(
                input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context", "relationship_context"],
                template=THOUGHT_TEMPLATE
            )
            self.thought_chain = LLMChain(llm=self.llm, prompt=thought_prompt, verbose=self.verbose)
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Thought chain initialized.{BColors.ENDC}")

        if not self.action_chain:
            action_prompt = PromptTemplate(
                 input_variables=["agent_name", "agent_traits", "current_time", "agent_status", "observation", "memory_context", "relationship_context"],
                template=ACTION_TEMPLATE
            )
            self.action_chain = LLMChain(llm=self.llm, prompt=action_prompt, verbose=self.verbose)
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Action chain initialized.{BColors.ENDC}")

        if not self.status_update_chain:
             status_update_prompt = PromptTemplate(
                 input_variables=["agent_name", "agent_traits", "previous_status", "action_taken"],
                 template=STATUS_UPDATE_TEMPLATE
             )
             self.status_update_chain = LLMChain(llm=self.llm, prompt=status_update_prompt, verbose=self.verbose)
             print(f"{BColors.DIM}DEBUG (Agent {self.name}): Status update chain initialized.{BColors.ENDC}")

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(ENTITY_EXTRACTION_TEMPLATE)
        try:
            entity = self.chain(prompt).run(agent_name=self.name, observation=observation).strip()
            if self.name.lower() in entity.lower() or "myself" in entity.lower() or "me" in entity.lower():
                print(f"{BColors.WARNING}WARN (Agent {self.name}): Entity extraction returned the agent itself. Using fallback.{BColors.ENDC}", flush=True)
                return "the other person in the observation"
            if "no other entity" in entity.lower() or "no entity" in entity.lower():
                print(f"{BColors.DIM}DEBUG (Agent {self.name}): No entity found in observation.{BColors.ENDC}", flush=True)
                return "the environment or situation"
            entity = entity.replace("The main entity is", "").replace("Main entity:", "").strip()
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Identified entity: '{entity}' from observation{BColors.ENDC}", flush=True)
            return entity
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to extract entity: {e}. Using fallback.{BColors.ENDC}", flush=True)
            return "the other person or entity in the observation"

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(ENTITY_ACTION_TEMPLATE)
        try:
            action = self.chain(prompt).run(entity=entity_name, observation=observation).strip()
            action = action.replace(f"{entity_name} is", "").strip()
            if action.startswith("is "):
                action = action[3:]
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Identified action: '{entity_name} is {action}'{BColors.ENDC}", flush=True)
            return action
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to extract action: {e}. Using fallback.{BColors.ENDC}", flush=True)
            return "present in the scene"

    def summarize_related_memories(self, observation: str, now: Optional[datetime] = None) -> str:
        current_time = now or datetime.now()
        if (self.cached_relationship_context is not None and
            self.cached_relationship_time is not None and
            (current_time - self.cached_relationship_time).total_seconds() < 1.0):
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Using cached relationship context{BColors.ENDC}", flush=True)
            return self.cached_relationship_context

        entity_name = self._get_entity_from_observation(observation)
        if entity_name in ["the environment or situation", "no other entity"]:
            default_context = "No specific entities to establish a relationship with."
            self.cached_relationship_context = default_context
            self.cached_relationship_time = current_time
            return default_context

        entity_action = self._get_entity_action(observation, entity_name)
        prompt = PromptTemplate.from_template(RELATIONSHIP_SUMMARY_TEMPLATE)

        try:
            relevant_memories = self.memory.fetch_memories(entity_name, now=now)
            if not relevant_memories:
                broader_queries = [
                    f"{entity_name}", f"relationship with {entity_name}",
                    f"opinion about {entity_name}", f"knowledge of {entity_name}"
                ]
                for query in broader_queries:
                    memories = self.memory.fetch_memories(query, now=now)
                    if memories:
                        relevant_memories.extend(memories)
                        break
            memory_str = "\n".join([f"- {mem.page_content}" for mem in relevant_memories]) if relevant_memories else "No specific memories about this entity."
            relationship_context = self.chain(prompt).run(
                entity_name=entity_name, entity_action=entity_action, relevant_memories=memory_str
            ).strip()
            if not relationship_context or relationship_context.lower() == "none" or "no relationship" in relationship_context.lower():
                relationship_context = f"You have no prior relationship with {entity_name}. This appears to be your first encounter."
            print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated relationship context: {relationship_context}{BColors.ENDC}", flush=True)
            self.cached_relationship_context = relationship_context
            self.cached_relationship_time = current_time
            return relationship_context
        except Exception as e:
            print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to summarize related memories: {e}. Using fallback.{BColors.ENDC}", flush=True)
            fallback = f"No specific relationship information available about {entity_name}."
            self.cached_relationship_context = fallback
            self.cached_relationship_time = current_time
            return fallback

    def _fetch_context(self, observation: str, now: Optional[datetime] = None) -> Tuple[str, str]:
        if not self.memory:
             raise ValueError("Agent memory is not initialized.")
        original_k = -1
        retriever = self.memory.memory_retriever
        if hasattr(retriever, 'k'):
            original_k = retriever.k
        try:
            relevant_memories: List[Document] = self.memory.fetch_memories(observation, now)
            memory_context = "\n".join([f"- {m.page_content.strip()}" for m in relevant_memories])
            if not memory_context.strip():
                memory_context = "No relevant memories."
        finally:
             if original_k != -1 and hasattr(retriever, 'k'):
                retriever.k = original_k
        current_time_str = (now or datetime.now()).strftime("%B %d, %Y, %I:%M:%S %p")
        return memory_context, current_time_str

    def get_summary(self, force_refresh: bool = False, now: Optional[datetime] = None) -> str:
        current_time = now or datetime.now()
        if (not force_refresh and
            self.cached_summary is not None and
            self.cached_summary_time is not None and
            (current_time - self.cached_summary_time).total_seconds() < 1.0):
            return self.cached_summary
        summary = super().get_summary(force_refresh=force_refresh, now=current_time)
        self.cached_summary = summary
        self.cached_summary_time = current_time
        return summary

    def _decide_reaction_type(self, observation: str, now: Optional[datetime] = None) -> str:
        self._initialize_chains()
        now = now or datetime.now()
        memory_context, current_time_str = self._fetch_context(observation, now)
        # agent_summary = self.get_summary(now=now, force_refresh=False) # Agent summary not used in decision prompt

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
        if not self.memory:
             raise ValueError("Agent memory is not initialized before generating reaction.")
        self.cached_summary = None
        self.cached_summary_time = None
        self.cached_relationship_context = None
        self.cached_relationship_time = None
        self._initialize_chains()
        call_time = now or datetime.now()
        
        relationship_context = self.summarize_related_memories(observation, call_time)
        reaction_type = self._decide_reaction_type(observation, call_time)

        observation_poignancy = 0
        add_observation_memory = False
        reflection_enabled = self.memory.reflection_threshold is not None and self.memory.reflection_threshold > 0

        if reflection_enabled:
            try:
                 if hasattr(self.memory, 'score_memory_importance'):
                     observation_poignancy = self.memory.score_memory_importance(observation)
                 else:
                    print(f"{BColors.WARNING}WARN (Agent {self.name}): memory.score_memory_importance not found. Using fallback scoring.{BColors.ENDC}")
                    poignancy_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(POIGNANCY_SCORING_FALLBACK_TEMPLATE))
                    try:
                       poignancy_result = poignancy_chain.run(observation=observation).strip()
                       observation_poignancy = int(poignancy_result)
                    except ValueError:
                       print(f"{BColors.WARNING}WARN (Agent {self.name}): Fallback poignancy scoring failed to parse int from '{poignancy_result}'. Defaulting to 3.{BColors.ENDC}")
                       observation_poignancy = 3
                 add_observation_memory = observation_poignancy >= self.memory.reflection_threshold
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Observation poignancy rated: {observation_poignancy} (Threshold: {self.memory.reflection_threshold}, AddMem: {add_observation_memory}){BColors.ENDC}", flush=True)
            except Exception as e:
                 print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to score observation poignancy: {e}. Defaulting importance based on reaction.{BColors.ENDC}", flush=True)
                 observation_poignancy = 3
                 add_observation_memory = reaction_type != "IGNORE"
        else:
            add_observation_memory = reaction_type != "IGNORE" # Add if any reaction other than IGNORE
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Reflection disabled. AddMem based on reaction: {add_observation_memory}{BColors.ENDC}")

        reaction_output = ""
        if reaction_type == "IGNORE":
            print(f"{BColors.DIM}DEBUG (Agent {self.name}): Chose to IGNORE.{BColors.ENDC}", flush=True)
            if add_observation_memory and reflection_enabled: # Only add if poignant and reflection is on
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Adding poignant observation to memory despite IGNORE.{BColors.ENDC}", flush=True)
                 self.memory.add_memory(observation, now=call_time)
            return (add_observation_memory, reaction_output) # add_observation_memory flag from poignancy check
        else:
             memory_context, current_time_str = self._fetch_context(observation, call_time)
             # agent_summary = self.get_summary(now=call_time, force_refresh=False) # Not used in thought/action prompts

             if reaction_type == "THINK":
                 try:
                     thought_text = self.thought_chain.run(
                         agent_name=self.name, agent_traits=self.traits, current_time=current_time_str,
                         agent_status=self.status, observation=observation, memory_context=memory_context,
                         relationship_context=relationship_context,
                     ).strip()
                     print(f"{BColors.OKBLUE}DEBUG (Agent {self.name}): Generated thought: {thought_text}{BColors.ENDC}", flush=True)
                     self.memory.add_memory(f"(Internal thought) {thought_text}", now=call_time)
                     reaction_output = f"THINK: {thought_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during thought chain: {e}{BColors.ENDC}", flush=True)
                     reaction_output = "THINK: (Error generating thought)"
                     self.memory.add_memory("(Internal thought) Error generating thought.", now=call_time)

             elif reaction_type == "DO":
                 try:
                     action_text = self.action_chain.run(
                          agent_name=self.name, agent_traits=self.traits, current_time=current_time_str,
                          agent_status=self.status, observation=observation, memory_context=memory_context,
                          relationship_context=relationship_context,
                     ).strip()
                     print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Generated action: {action_text}{BColors.ENDC}", flush=True)
                     self.memory.add_memory(f"(Action) {action_text}", now=call_time)
                     try:
                        updated_status = self.status_update_chain.run(
                            agent_name=self.name, agent_traits=self.traits,
                            previous_status=self.status, action_taken=action_text,
                        ).strip().split('\n')[0].replace('"', '').replace("'", '').strip()
                        if updated_status:
                            print(f"{BColors.OKGREEN}DEBUG (Agent {self.name}): Status updated from '{self.status}' to '{updated_status}'{BColors.ENDC}", flush=True)
                            self.status = updated_status
                     except Exception as e_status:
                        print(f"{BColors.WARNING}WARN (Agent {self.name}): Failed to update status via LLM: {e_status}. Using fallback.{BColors.ENDC}", flush=True)
                        self.status = f"Just {action_text}" # Simple fallback
                     reaction_output = f"DO: {action_text}"
                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Exception during action chain: {e}{BColors.ENDC}", flush=True)
                     reaction_output = "DO: (Error generating action)"
                     self.memory.add_memory("(Action) Error generating action.", now=call_time)

             elif reaction_type == "SAY":
                 try:
                    print(f"{BColors.DIM}DEBUG (Agent {self.name}): Calling super().generate_dialogue_response for SAY.{BColors.ENDC}", flush=True)
                    # The base class's generate_dialogue_response adds both observation and response to memory
                    # The boolean it returns refers to the poignancy of the observation it calculated internally
                    # We rely on our own `add_observation_memory` flag for consistency.
                    _base_dialogue_obs_important, dialogue_text = super().generate_dialogue_response(observation, call_time)
                    print(f"{BColors.HEADER}DEBUG (Agent {self.name}): Generated dialogue: {dialogue_text}{BColors.ENDC}", flush=True)
                    reaction_output = f"SAY: {dialogue_text}"
                    # Note: super().generate_dialogue_response already adds observation and the dialogue to memory.
                    # We don't need to add the observation again here if it was a SAY action.
                    # However, our `add_observation_memory` flag is based on our poignancy check.
                    # If the base class didn't add the observation due to its own logic, but ours says it's important,
                    # we should ensure it's added. But `GenerativeAgent` adds observation *before* dialogue.
                    # So, the observation is likely already added. This part might need refinement if discrepancies occur.
                    # For now, we assume base class handles memory for SAY correctly.
                    # We override the add_observation_memory flag for SAY to false here because super() handles it.
                    add_observation_memory = False # Prevent double-adding of observation for SAY

                 except Exception as e:
                     print(f"{BColors.FAIL}ERROR (Agent {self.name}): Failed during SAY generation step (super call): {e}. Defaulting to THINK.{BColors.ENDC}", flush=True)
                     reaction_output = "THINK: (Failed to formulate a verbal response)"
                     self.memory.add_memory("(Internal thought) Failed to formulate a verbal response.", now=call_time)

             if add_observation_memory: # This flag is now true only if poignancy check passed AND not a SAY action (where super() handles it)
                 print(f"{BColors.DIM}DEBUG (Agent {self.name}): Adding observation memory (Poignancy: {observation_poignancy}, Reaction: {reaction_type}).{BColors.ENDC}", flush=True)
                 self.memory.add_memory(observation, now=call_time)

        final_importance_flag = add_observation_memory or (reaction_type == "SAY" and reflection_enabled and observation_poignancy >= self.memory.reflection_threshold)
        return (final_importance_flag, reaction_output)

    def generate_dialogue_response(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        observation_important, reaction_string = self.generate_reaction(observation, now)
        if reaction_string.startswith("SAY:"):
            dialogue = reaction_string[len("SAY:"):].strip()
            return observation_important, dialogue
        else:
            # If it wasn't a SAY reaction, it means no dialogue was generated.
            # The observation_important flag still reflects the poignancy of the original observation.
            return observation_important, ""
