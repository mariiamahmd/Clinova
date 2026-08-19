from src.retrieval import hybrid_search
from src.reranking import rerank_results

test_questions = [
    {'question': 'How can I reduce my risk of developing skin cancer?', 
     'type': 'direct', 
     'expected_doc': 'skin_prevention', 
     'section': 3, 
     'pages': [11, 12, 13, 14, 15], 
     'relevant_chunks': ['skin_prevention_s3_c14', 'skin_cancer_s2_c8', 'skin_cancer_s2_c9'], 
     'reason': 'The chunk contains the Prevention section and directly lists avoiding getting burnt, staying in the shade during the middle of the day, wearing protective clothing, and using high-SPF products.'}, 

     {'question': 'Which people should take extra care to avoid skin damage and skin cancer?',
       'type': 'direct', 
       'expected_doc': 
       'skin_cancer', 
       'section': 2, 
       'pages': [6, 7, 8, 9, 10], 
       'relevant_chunks': ['skin_cancer_s2_c9', 'skin_cancer_s2_c10', 'skin_prevention_s2_c9'], 
       'reason': 'The chunk contains the At-risk groups section and lists people who should take extra care, including children, people who burn easily, people with fair skin, people with many moles, immunosuppressed people, and others.'}, 

       {'question': 'What are the main risk factors for developing skin cancer?', 
        'type': 'direct', 
        'expected_doc': 'skin_prevention', 
        'section': 3, 
        'pages': [11, 12, 13, 14, 15], 
        'relevant_chunks': ['skin_prevention_s3_c13', 'skin_prevention_s3_c14', 'skin_cancer_s2_c9', 'skin_cancer_s2_c10'],
          'reason': 'This chunk contains the Risk factors section, including UV radiation, age and sex, sunburn, occupation, personal and family history, physical characteristics, regional variation, and socioeconomic status.'},

     {'question': 'What are the main ways to protect the skin from excessive sunlight exposure?',
       'type': 'direct', 
       'expected_doc': 'skin_cancer', 
       'section': 2,
         'pages': [6, 7, 8, 9, 10],
         'relevant_chunks': ['skin_cancer_s2_c9', 'skin_prevention_s3_c14'],
           'reason': 'These chunks discuss approaches to protecting the skin, suncream, and groups that need extra protection.'},

            {'question': 'What can someone do to lower their chances of getting skin cancer?', 
              'type': 'paraphrased', 
              'expected_doc': 'skin_prevention',
                'section': 3,
                  'pages': [11, 12, 13, 14, 15], 
                  'relevant_chunks': ['skin_prevention_s3_c14', 'skin_cancer_s2_c8', 'skin_cancer_s2_c9'], 
                  'reason': 'This paraphrases the prevention information describing ways to reduce the risk of skin cancer.'},

                    {'question': 'Who is more vulnerable to skin damage caused by sunlight?', 
                     'type': 'paraphrased',
                       'expected_doc': 'skin_cancer', 
                       'section': 2, 
                       'pages': [6, 7, 8, 9, 10], 
                       'relevant_chunks': ['skin_cancer_s2_c9', 'skin_cancer_s2_c10', 'skin_prevention_s2_c9'],
                         'reason': 'The chunk lists multiple at-risk groups, including children, people who burn rather than tan, people with fair skin, people with many moles, and immunosuppressed people.'}, 

                         {'question': "Which personal characteristics can increase someone's likelihood of developing skin cancer?", 
                          'type': 'paraphrased',
                            'expected_doc': 'skin_prevention', 
                            'section': 3,
                              'pages': [11, 12, 13, 14, 15],
                                'relevant_chunks': ['skin_prevention_s3_c13', 'skin_prevention_s3_c14', 'skin_cancer_s2_c9'], 
                                'reason': 'The risk-factor chunk specifically discusses physical characteristics such as fair skin that burns easily, many moles or freckles, and red or fair hair or light-coloured eyes.'}, 

                                {'question': 'How can people reduce the harmful effects of too much sun exposure?', 
                                 'type': 'paraphrased', 
                                 'expected_doc': 'skin_prevention', 
                                 'section': 3, 
                                 'pages': [11, 12, 13, 14, 15], 
                                 'relevant_chunks': ['skin_prevention_s3_c14', 'skin_prevention_s3_c15', 'skin_cancer_s2_c8', 'skin_cancer_s2_c9'], 
                                 'reason': 'The chunks explain practical sun-protection measures such as avoiding sunburn, using shade, wearing protective clothing and using high-SPF products. They also explain that sun protection should be balanced with the benefits of sunlight, including physical activity and vitamin D.'}, 

                                 {'question': 'What does PHIAC stand for in the skin cancer prevention guidance?',
                                   'type': 'abbreviation', 
                                   'expected_doc': 'skin_prevention', 
                                   'section': 2,
                                     'pages': [6, 7, 8, 9, 10], 
                                   'relevant_chunks': ['skin_prevention_s2_c4'], 
                                   'reason': 'The chunk gives the full name Public Health Interventions Advisory Committee and explains its role in developing the guidance.'},

                                     {'question': 'What does UV mean in the context of skin cancer risk?', 
                                      'type': 'abbreviation',
                                        'expected_doc': 'skin_prevention', 
                                        'section': 3, 
                                        'pages': [11, 12, 13, 14, 15], 
                                        'relevant_chunks': ['skin_prevention_s3_c13'], 
                                        'reason': 'The risk-factor chunk discusses ultraviolet (UV) radiation as the leading cause of skin cancer.'},

                                          {'question': 'Before what age is frequent exposure to artificial UV light from sunbeds identified as an increased risk?', 
                                           'type': 'threshold', 
                                           'expected_doc': 'skin_cancer', 
                                           'section': 2, 
                                           'pages': [6, 7, 8, 9, 10], 
                                           'relevant_chunks': ['skin_cancer_s2_c10'], 
                                           'reason': 'The chunk states that people frequently exposed to artificial UV light, including from sunbeds, before the age of 25 are an increased-risk group.'}, 

                                           {'question': 'Within how many days should a diagnosis of suspected melanoma or another specified skin cancer be confirmed or ruled out after referral?', 
                                            'type': 'threshold', 
                                            'expected_doc': 'skin_cancer', 
                                            'section': 3, 
                                            'pages': [11, 12, 13, 14, 15], 
                                            'relevant_chunks': ['skin_cancer_s3_c12', 'skin_cancer_s3_c13'], 
                                            'reason': 'The quality statement specifies that suspected melanoma, squamous cell carcinoma or a rare skin cancer should have the diagnosis confirmed or ruled out within 28 days of referral.'}, 

                                            {'question': 'Within how many days should the first treatment be provided after the decision to treat?', 
                                             'type': 'threshold', 
                                             'expected_doc': 'skin_cancer', 
                                             'section': 3, 
                                             'pages': [11, 12, 13, 14, 15], 
                                             'relevant_chunks': ['skin_cancer_s3_c13'], 
                                             'reason': 'The chunk contains the quality-standard timing for the period from the decision to treat to first treatment (31 days).'}, 

                                             {'question': 'What is identified as the leading cause of skin cancer?', 
                                              'type': 'factual', 
                                              'expected_doc': 'skin_prevention', 
                                              'section': 3, 
                                              'pages': [11, 12, 13, 14, 15], 
                                              'relevant_chunks': ['skin_prevention_s3_c13'], 
                                              'reason': 'The Risk factors section explicitly identifies exposure to ultraviolet radiation as the leading cause of skin cancer.'}, 

                                              {'question': 'Which types of workers are mentioned as being particularly at risk of skin cancer?', 
                                               'type': 'factual', 
                                               'expected_doc': 'skin_prevention', 
                                               'section': 3, 
                                               'pages': [11, 12, 13, 14, 15], 
                                               'relevant_chunks': ['skin_prevention_s3_c13', 'skin_prevention_s3_c14', 'skin_cancer_s2_c9', 'skin_cancer_s2_c10'], 
                                               'reason': 'The chunk lists outdoor workers and people involved in outdoor sports, including construction workers, cricketers, golfers, farmers, gardeners, military personnel and postal workers.'}, 

                                               {'question': 'Which groups of people who spend a lot of time in the sun are identified as being at increased risk of skin cancer?', 
                                                'type': 'factual', 
                                                'expected_doc': 'skin_cancer', 
                                                'section': 2, 
                                                'pages': [6, 7, 8, 9, 10], 
                                                'relevant_chunks': ['skin_prevention_s3_c14', 'skin_cancer_s2_c9'], 
                                                'reason': 'The chunk identifies outdoor workers and people with outdoor hobbies, such as sailing or golf, as groups who spend a lot of time in the sun and are at increased risk.'}, 

                                                {'question': 'What are some physical characteristics associated with a higher risk of skin cancer?', 
                                                 'type': 'factual', 
                                                 'expected_doc': 'skin_prevention', 
                                                 'section': 3, 
                                                 'pages': [11, 12, 13, 14, 15], 
                                                 'relevant_chunks': ['skin_prevention_s2_c8', 'skin_prevention_s3_c15', 'skin_cancer_s2_c8'], 
                                                 'reason': 'The chunk mentions fair skin that burns easily, many moles or freckles, red or fair hair, and light-coloured eyes.'}, 

                                                 {'question': 'Why does the guidance recommend balancing sun protection with the benefits of sunlight?', 
                                                  'type': 'comprehension', 
                                                  'expected_doc': 'skin_prevention', 
                                                  'section': 3, 
                                                  'pages': [11, 12, 13, 14, 15], 
                                                  'relevant_chunks': ['skin_prevention_s3_c10', 'skin_prevention_s3_c11'], 
                                                  'reason': 'These chunks explain that sunlight provides benefits such as wellbeing, vitamin D synthesis and opportunities for physical activity, while reducing sun exposure too much may reduce physical activity and increase vitamin D deficiency.'}, 

                                                  {'question': 'Why should skin cancer prevention activities not discourage outdoor physical activity?', 
                                                   'type': 'comprehension', 
                                                   'expected_doc': 'skin_prevention', 
                                                   'section': 2, 
                                                   'pages': [6, 7, 8, 9, 10], 
                                                   'relevant_chunks': ['skin_prevention_s2_c6'], 
                                                   'reason': 'The guidance explains that prevention activities may inadvertently reduce physical activity if people try to avoid sun exposure completely, so sensible skin protection should be encouraged instead.'}, 

                                                   {'question': 'What is the recommended treatment for bacterial pneumonia?', 
                                                    'type': 'out_of_scope', 
                                                    'expected_doc': None, 
                                                    'section': None, 
                                                    'pages': [], 
                                                    'relevant_chunks': [], 
                                                    'reason': 'The indexed documents concern skin cancer, skin cancer prevention, sunlight exposure and related guidance. They do not provide evidence about treating bacterial pneumonia.'},

                                                     {'question': 'What is the normal blood pressure range for a healthy adult?', 
                                                      'type': 'out_of_scope', 
                                                      'expected_doc': None, 
                                                      'section': None, 
                                                      'pages': [], 
                                                      'relevant_chunks': [],
                                                     'reason': 'Blood pressure is outside the scope of the indexed skin cancer guidance, so the system should not use unrelated skin-cancer chunks to answer this question.'}]

def precision_at_k(results, relevant_chunks, k):
    if not relevant_chunks:
        return None

    retrieved_ids = [
        result["chunk_id"]
        for result in results[:k]
    ]

    relevant_count = sum(
        chunk_id in relevant_chunks
        for chunk_id in retrieved_ids
    )

    return relevant_count / k


def recall_at_k(results, relevant_chunks, k):
    if not relevant_chunks:
        return None

    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    relevant_ids = set(relevant_chunks)

    relevant_found = len(
        retrieved_ids & relevant_ids
    )

    return relevant_found / len(relevant_ids)


def evaluate_retriever(test_questions, k=10):

    precision_scores = []
    recall_scores = []

    details = []

    for test in test_questions:
        candidates = hybrid_search(test["question"],k,k,k)

        if test["type"] == "out_of_scope":
            precision = None
            recall = None
            relevant = len(candidates) == 0

        else:

            precision = precision_at_k(candidates,test["relevant_chunks"],k)
            recall = recall_at_k(candidates,test["relevant_chunks"],k)

            relevant = recall > 0

        if precision is not None:
            precision_scores.append(precision)

        if recall is not None:
            recall_scores.append(recall)

        details.append({
            "question":
                test["question"],

            "type":
                test["type"],

            "expected":
                test["relevant_chunks"],

            "retrieved":
                [
                    x["chunk_id"]
                    for x in candidates
                ],

            "precision":
                precision,

            "recall":
                recall,

            "relevant_found":
                relevant
        })


    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
        if precision_scores
        else 0
    )

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
        if recall_scores
        else 0
    )


    metrics = {
        f"Precision@{k}":
            average_precision,

        f"Recall@{k}":
            average_recall
    }


    return metrics, details


def evaluate_reranker(test_questions,candidate_k=10,final_k=2):
    precision_scores = []
    recall_scores = []
    details = []

    for test in test_questions:

        candidates = hybrid_search(test["question"],candidate_k,candidate_k,candidate_k)

        reranked = rerank_results(test["question"],candidates,final_k)

        if test["type"] == "out_of_scope":
            precision = None
            recall = None
            relevant = len(reranked) == 0
        else:

            precision = precision_at_k(reranked,test["relevant_chunks"],final_k)
            recall = recall_at_k(reranked,test["relevant_chunks"],final_k)
            relevant = recall > 0

        if precision is not None:
            precision_scores.append(precision)

        if recall is not None:
            recall_scores.append(recall)

        details.append({

            "question":
                test["question"],

            "type":
                test["type"],

            "expected":
                test["relevant_chunks"],

            "before_reranking":
                [
                    x["chunk_id"]
                    for x in candidates
                ],

            "after_reranking":
                [
                    x["chunk_id"]
                    for x in reranked
                ],

            "precision":
                precision,

            "recall":
                recall,

            "relevant_found":
                relevant
        })


    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
        if precision_scores
        else 0
    )

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
        if recall_scores
        else 0
    )


    metrics = {
        f"Precision@{final_k}":
            average_precision,

        f"Recall@{final_k}":
            average_recall
    }


    return metrics, details