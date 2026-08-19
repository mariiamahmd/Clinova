import streamlit as st

from src.evaluation import (
    test_questions,
    evaluate_retriever,
    evaluate_reranker
)


st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="📊",
    layout="wide"
)


st.title("📊 RAG Evaluation Dashboard")

st.write(
    "Internal evaluation dashboard for system performance."
)


candidate_k = st.slider(
    "Retriever Top-K",
    min_value=5,
    max_value=20,
    value=10
)


final_k = st.slider(
    "Reranker Final-K",
    min_value=1,
    max_value=5,
    value=2
)


st.write(
    f"Number of test questions: "
    f"**{len(test_questions)}**"
)


if st.button(
    "▶ Run Evaluation",
    type="primary"
):

    with st.spinner(
        "Evaluating retriever..."
    ):

        (
            retriever_metrics,
            retriever_details
        ) = evaluate_retriever(
            test_questions,
            k=candidate_k
        )


    with st.spinner(
        "Evaluating reranker..."
    ):

        (
            reranker_metrics,
            reranker_details
        ) = evaluate_reranker(
            test_questions,
            candidate_k=candidate_k,
            final_k=final_k
        )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
    f"Retriever Precision@{candidate_k}",
    f"{retriever_metrics[f'Precision@{candidate_k}']:.3f}"
)

        st.metric(
            f"Retriever Recall@{candidate_k}",
            f"{retriever_metrics[f'Recall@{candidate_k}']:.3f}"
        )

    with col2:

        st.metric(
        f"Reranker Precision@{final_k}",
        f"{reranker_metrics[f'Precision@{final_k}']:.3f}"
    )


    st.divider()


    # ========================================================
    # QUESTIONS
    # ========================================================

    st.subheader(
        "Question Evaluation"
    )


    for i, (
        retriever_item,
        reranker_item
    ) in enumerate(
        zip(
            retriever_details,
            reranker_details
        ),
        start=1
    ):

        with st.expander(
            f"Question {i}: "
            f"{retriever_item['question']}"
        ):

            expected = set(
                retriever_item["expected"]
            )


            # ------------------------------------------------
            # EXPECTED
            # ------------------------------------------------

            st.markdown(
                "### Expected Relevant Chunks"
            )

            for chunk in expected:

                st.write(
                    f"`{chunk}`"
                )


            # ------------------------------------------------
            # RETRIEVER
            # ------------------------------------------------

            st.markdown(
                f"### Retriever — Top {candidate_k}"
            )


            for rank, chunk in enumerate(
                retriever_item["retrieved"],
                start=1
            ):

                if chunk in expected:

                    st.success(
                        f"{rank}. `{chunk}` ✓ Relevant"
                    )

                else:

                    st.write(
                        f"{rank}. `{chunk}` "
                        f"✗ Not relevant"
                    )


            st.info(
                f"Precision@{candidate_k}: "
                f"**{retriever_item['precision']:.3f}**"
            )


            st.divider()


            # ------------------------------------------------
            # RERANKER
            # ------------------------------------------------

            st.markdown(
                f"### Reranker — Top {final_k}"
            )


            for rank, chunk in enumerate(
                reranker_item["after_reranking"],
                start=1
            ):

                if chunk in expected:

                    st.success(
                        f"{rank}. `{chunk}` ✓ Relevant"
                    )

                else:

                    st.error(
                        f"{rank}. `{chunk}` "
                        f"✗ Not relevant"
                    )


            st.info(
                f"Precision@{final_k}: "
                f"**{reranker_item['precision']:.3f}**"
            )