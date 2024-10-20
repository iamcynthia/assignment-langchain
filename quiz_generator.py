from abc import ABC, abstractmethod
from typing import List
from schema import Quiz, Quizzes

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class HistoryQuizGenerator(ABC):
    """
    A quiz generator generating history quiz based on the given content and keywords.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    @abstractmethod
    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
        """
        Create a quiz including its question, and four options. Based on the given content and keywords.
        Please use GPT-3.5 with LangChain to implement the function.
        """
        pass

    @abstractmethod
    def create_quizzes(
        self, content: str, topics: List[str], num_quizzes: int
    ) -> Quizzes:
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

class CreateHistoryQuiz(HistoryQuizGenerator):
    def create_quiz(self, content: str, keywords: List[str]) -> Quiz:
                
        messages = [
            SystemMessage(
                content="You are a history teacher who create a historic multiple-choice question. The question should have 4 options with only one correct answer. Based on the given content and keywords."
            ),
            HumanMessage(
                content=f"Content: {content}\n Keyword: {', '.join(keywords)}"
            )
        ]

        response = self.model.invoke(messages)
        quiz_data = response.content
        return quiz_data
    
    def create_quizzes(self, content: str, topics: List[str], num_quizzes: int) -> Quizzes:
        return super().create_quizzes(content, topics, num_quizzes)
    

class CreateMathQuiz(MathQuizGenerator):
    def create_quiz(self) -> Quiz:
        messages = [
            SystemMessage(
                content="You are a math teacher who create a 2 variables linear equation math word question. The question should have 4 options with only one correct answer. Write down the calculation formula in each option."
            )
        ]

        response = self.model.invoke(messages)
        quiz_data = response.content
        return quiz_data
    
    def create_quizzes(self, num_quizzes: int) -> Quizzes:
        return super().create_quizzes(num_quizzes)