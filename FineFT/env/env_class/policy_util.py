def get_close_element(intended_action, acaiable_action_list):
    return min(acaiable_action_list, key=lambda x: abs(x - intended_action))
