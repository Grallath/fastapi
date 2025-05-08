# File: prompts.py

# --- AutonomousGenerativeAgent Prompts ---

DECISION_TEMPLATE = (
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

THOUGHT_TEMPLATE = (
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

ACTION_TEMPLATE = (
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

STATUS_UPDATE_TEMPLATE = (
    "You are {agent_name}.\n"
    "Your core characteristics are: {agent_traits}\n"
    "Your previous status was: {previous_status}\n"
    "You just performed the action: {action_taken}\n"
    "Based on this action, what is your concise, updated status? Describe it in the first person (e.g., 'Standing alert.', 'Sitting and observing.')."
    "\nUpdated Status:"
)

ENTITY_EXTRACTION_TEMPLATE = (
    "In the following observation, identify the main entity or person OTHER THAN {agent_name} who is being observed. "
    "If there are multiple entities, identify the most prominent one. "
    "If there is no entity other than {agent_name}, respond with 'no other entity'.\n\n"
    "Observation: {observation}\n\n"
    "Main entity (not {agent_name}):"
)

ENTITY_ACTION_TEMPLATE = (
    "Based on the following observation, what is {entity} doing? Describe their actions concisely.\n\n"
    "Observation: {observation}\n\n"
    "What {entity} is doing:"
)

RELATIONSHIP_SUMMARY_TEMPLATE = (
    "Based on your memories, what is your relationship or knowledge about {entity_name}?\n"
    "Consider:\n"
    "1. Have you met {entity_name} before?\n"
    "2. Do you have any history with {entity_name}?\n"
    "3. Do you have any feelings or opinions about {entity_name}?\n"
    "4. Is there anything notable about {entity_name}?\n\n"
    "Context from your memories:\n{relevant_memories}\n\n"
    "Current observation: {entity_name} is {entity_action}\n\n"
    "Relationship with {entity_name} (be concise, if no relationship exists, state that clearly):"
)

POIGNANCY_SCORING_FALLBACK_TEMPLATE = (
    "Rate the poignancy of this observation on a scale of 1 to 10 (integer): {observation}\nRating:"
)
