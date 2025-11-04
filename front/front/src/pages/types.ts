// lo que devuelve /generate
export type PromptResponse = {
    response: string;
    model: string;
    };

    // lo que devuelve /evaluateAnswer
    export type EvaluateAnswerResponse = {
    response: {
        rating: number;       // 0..100
        answer: string;       // respuesta recomendada por el LLM
    };
    model: string;
};
