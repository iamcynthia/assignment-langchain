from abc import ABC, abstractmethod
from typing import List
from schema import Quiz, Quizzes
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class HistoryQuizGenerator(ABC):
    """
    A quiz generator generating history quiz based on the given content and keywords.
    """

    @abstractmethod
    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
        """
        Create a quiz including its question, and four options. Based on the given content and keywords.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass

    @abstractmethod
    def create_quizzes(self, content: str, topics: List[str], num_quizzes: int) -> Quizzes:
        """
        Create multiple quizzes.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass


class MathQuizGenerator(ABC):
    """
    A quiz generator generating math word quiz which involves a linear equation with two variables.
    """

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    @abstractmethod
    def create_quiz(self) -> Quiz:
        """
        Create a math word quiz which involves a linear equation with two variables.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass

    @abstractmethod
    def create_quizzes(self, num_quizzes: int) -> Quizzes:
        """
        Create multiple quizzes.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass


class History(HistoryQuizGenerator):
    def __init__(self) -> None:
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a history teacher who generate a history multiple-choice question. The question should have 4 options with only one correct answer.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Generate a question about the given content: {content}, and the given keywords are: {', '.join(keywords)}"
                    )
                ]
            }
        )

        return response.content

    def create_quizzes(self, content: str, topics: List[str], num_quizzes: int) -> Quizzes:
        pass


class Math(MathQuizGenerator):
    def __init__(self) -> None:
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    def create_quiz(self) -> Quiz:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a math teacher who generate math word questions. The question should have 4 options with only one correct answer.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Generate a math word question which includes a linear equation with 2 variables. Besides, write down the calculation formula with explanations in the options."
                    )
                ]
            }
        )

        return response.content

    def create_quizzes(self, num_quizzes: int) -> Quizzes:
        pass
