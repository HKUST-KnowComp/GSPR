'''
Label functions for the huggingface dataset.
item: dataframe row
Return a dictionary with:
    safety: "safe" or "unsafe"
    category: category of the item, e.g., "harassment", "hate", etc.
'''

import response_templates 


def get_label_fn(name, source, subset=None):
    """Get the label function based on the dataset source and subset."""

    # Mapping table for datasets and their label functions, keys in "source__subset" format
    name_to_label_fn = {
        'PKU-SafeRLHF_default_0': SafeRLHF0_label_fn,
        'PKU-SafeRLHF_default_1': SafeRLHF1_label_fn,
        'BeaverTails': BeaverTails_label_fn,
        'T2T':t2t_label_fn,
        'do_not_answer':NoAnswer_ladel_fn
        #'allenai/wildguardmix__wildguardtrain': wildguardmix_label_fn,
    }

    # Construct the key for the mapping table (source + subset, joined by double underscores)
    key = name
    # if subset:
    #     key += f"__{subset}"
    # Return the corresponding label function, raise exception if not found
    if key in name_to_label_fn:
        return name_to_label_fn[key]
    else:
        raise ValueError(f"Label function for {key} not found.")
    

def SafeRLHF0_label_fn(item):
    """Label function for WildGuardMix dataset.
    """
    category_dict = item['response_0_harm_category'] ## dict
    safety = item['is_response_0_safe'] ## bool
    category_count = 0
    category = ''
    # Unify safety label format: convert "unharmful" to "safe", "harmful" to "unsafe"
    if safety==True:
        safety = "safe"
        category = "not applicable"
    elif safety == False:
        safety = "unsafe"
        for label in category_dict:
            if category_dict[label] == True:
                category_count += 1
                category = category + '@@@' + label.lower()
        category = category.strip('@')
    else:
        raise ValueError(f"Safety label {safety} not recognized.")

    # Data validation: ensure safety label is in correct format
    assert safety in ["safe", "unsafe"], f"Safety label {safety} not recognized."

    if category_count > 1:
    #     raise ValueError(f"More than one category found for category_dict, skip this: {category_dict}")
    # Data validation: ensure category is in valid list (including "not applicable")
        category_list = category.split('@@@')
        for cat in category_list:
            assert cat in response_templates.category_list_PKU_SafeRLHF, f"Category {cat} not in category_list_PKU_SafeRLHF."
    else:
        assert category in response_templates.category_list_PKU_SafeRLHF + ['not applicable'], f"Category {category} not in category_list_PKU_SafeRLHF."

    return {
        'safety': safety,
        'category': category,
    }


def SafeRLHF1_label_fn(item):
    """Label function for WildGuardMix dataset.
    """
    category_dict = item['response_1_harm_category'] ## dict
    safety = item['is_response_1_safe'] ## bool
    category_count = 0
    category = ''
    # Unify safety label format: convert "unharmful" to "safe", "harmful" to "unsafe"
    if safety==True:
        safety = "safe"
        category = "not applicable"
    elif safety == False:
        safety = "unsafe"
        for label in category_dict:
            if category_dict[label] == True:
                category_count += 1
                category = category + '@@@' + label.lower()
        category = category.strip('@')
    else:
        raise ValueError(f"Safety label {safety} not recognized.")

    # Data validation: ensure safety label is in correct format
    assert safety in ["safe", "unsafe"], f"Safety label {safety} not recognized."

    if category_count > 1:
    #     raise ValueError(f"More than one category found for category_dict, skip this: {category_dict}")
    # Data validation: ensure category is in valid list (including "not applicable")
        category_list = category.split('@@@')
        for cat in category_list:
            assert cat in response_templates.category_list_PKU_SafeRLHF, f"Category {cat} not in category_list_BeaverTails."
    else:
        assert category in response_templates.category_list_PKU_SafeRLHF + ['not applicable'], f"Category {category} not in category_list_BeaverTails."

    return {
        'safety': safety,
        'category': category,
    }




def BeaverTails_label_fn(item):
    """Label function for BeaverTails dataset.
    """
    category_dict = item['category'] ## dict
    safety = item['is_safe'] ## bool
    category_count = 0
    category = ''
    # Unify safety label format: convert "unharmful" to "safe", "harmful" to "unsafe"
    if safety==True:
        safety = "safe"
        category = "not applicable"
    elif safety == False:
        safety = "unsafe"
        for label in category_dict:
            if category_dict[label] == True:
                category_count += 1
                category = category + '@@@' + label.lower()
        category = category.strip('@')
    else:
        raise ValueError(f"Safety label {safety} not recognized.")

    # Data validation: ensure safety label is in correct format
    assert safety in ["safe", "unsafe"], f"Safety label {safety} not recognized."

    if category_count > 1:
    #     raise ValueError(f"More than one category found for category_dict, skip this: {category_dict}")
    # Data validation: ensure category is in valid list (including "not applicable")
        category_list = category.split('@@@')
        for cat in category_list:
            assert cat in response_templates.category_list_BeaverTails, f"Category {cat} not in category_list_BeaverTails."
    else:
        assert category in response_templates.category_list_BeaverTails + ['not applicable'], f"Category {category} not in category_list_BeaverTails."

    return {
        'safety': safety,
        'category': category,
    }

def t2t_label_fn(item):
    safety = item['class']
    if safety == 'unsafe':
        category = item['category'].lower()
    elif 'safe' in safety:
        safety = 'safe'
        category = "not applicable"
    else:
        raise ValueError(f"Safety label {safety} not recognized.")
    
    return {
        'safety': safety,
        'category': category,
    }

def NoAnswer_ladel_fn(item):
    safety = item['ChatGLM2_harmful'] # bool
    if safety == 0:
        safety = 'safe'
        category = "not applicable"
    elif safety == 1:
        safety = 'unsafe'
        category = item['types_of_harm'].lower()
    return {
        'safety': safety,
        'category': category,
    }