"""
피드백 평가를 위한 헬퍼 모듈
사용자 요구사항과 도구 실행 결과를 비교하여 품질을 평가합니다.
"""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

class EvaluationCriteria(str, Enum):
    """평가 기준"""
    ACCURACY = "accuracy"        # 정확성
    COMPLETENESS = "completeness"  # 완성도
    RELEVANCE = "relevance"      # 관련성
    QUALITY = "quality"          # 품질
    USEFULNESS = "usefulness"    # 유용성

class FeedbackScore(BaseModel):
    """피드백 점수 모델"""
    overall_score: int  # 1-10 전체 점수
    criteria_scores: Dict[str, int]  # 기준별 점수
    reasoning: str  # 평가 이유
    improvements: List[str]  # 개선사항
    strengths: List[str]  # 장점

class FeedbackEvaluator:
    """피드백 평가기"""
    
    def __init__(self, llm=None, custom_criteria: Optional[List[str]] = None):
        self.llm = llm
        self.custom_criteria = custom_criteria or []
        self.default_criteria = [
            EvaluationCriteria.ACCURACY,
            EvaluationCriteria.COMPLETENESS, 
            EvaluationCriteria.RELEVANCE,
            EvaluationCriteria.QUALITY
        ]
    
    def evaluate_result(
        self, 
        user_request: str,
        tool_result: str,
        criteria: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> FeedbackScore:
        """결과를 평가하여 피드백 점수 생성"""
        
        evaluation_criteria = criteria or [c.value for c in self.default_criteria]
        
        if self.llm:
            return self._llm_evaluate(user_request, tool_result, evaluation_criteria, context)
        else:
            return self._rule_based_evaluate(user_request, tool_result, evaluation_criteria)
    
    def _llm_evaluate(
        self,
        user_request: str,
        tool_result: str, 
        criteria: List[str],
        context: Optional[str]
    ) -> FeedbackScore:
        """LLM을 사용한 평가"""
        
        context_text = f"\n컨텍스트: {context}" if context else ""
        criteria_text = ", ".join(criteria)
        
        evaluation_prompt = f"""
사용자 요청을 분석하고 도구 실행 결과가 얼마나 요구사항을 충족했는지 평가해주세요.

사용자 요청: {user_request}{context_text}

도구 실행 결과: {tool_result}

평가 기준: {criteria_text}

각 기준별로 1-10점 척도로 평가하고, 전체적인 점수를 매겨주세요:
1-2: 매우 부족함
3-4: 부족함  
5-6: 보통
7-8: 좋음
9-10: 매우 좋음

다음 JSON 형식으로 응답해주세요:
{{
    "overall_score": <1-10 사이의 전체 점수>,
    "criteria_scores": {{
        "{criteria[0] if criteria else 'accuracy'}": <점수>,
        "{criteria[1] if len(criteria) > 1 else 'completeness'}": <점수>,
        "{criteria[2] if len(criteria) > 2 else 'relevance'}": <점수>,
        "{criteria[3] if len(criteria) > 3 else 'quality'}": <점수>
    }},
    "reasoning": "<전체적인 평가 이유>",
    "improvements": ["<개선이 필요한 점1>", "<개선이 필요한 점2>"],
    "strengths": ["<잘된 점1>", "<잘된 점2>"]
}}
"""
        
        try:
            # LLM 호출 (실제 구현에서는 langchain chain 사용)
            from langchain_core.output_parsers import JsonOutputParser
            from langchain.schema.output_parser import StrOutputParser
            
            parser = JsonOutputParser()
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(evaluation_prompt)
                if hasattr(response, 'content'):
                    result = parser.parse(response.content)
                else:
                    result = parser.parse(str(response))
            else:
                # fallback to string parsing
                result = {
                    "overall_score": 5,
                    "criteria_scores": {c: 5 for c in criteria},
                    "reasoning": "LLM evaluation unavailable",
                    "improvements": ["Unable to perform detailed evaluation"],
                    "strengths": ["Result generated successfully"]
                }
            
            return FeedbackScore(
                overall_score=result.get("overall_score", 5),
                criteria_scores=result.get("criteria_scores", {}),
                reasoning=result.get("reasoning", ""),
                improvements=result.get("improvements", []),
                strengths=result.get("strengths", [])
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {str(e)}")
            return self._rule_based_evaluate(user_request, tool_result, criteria)
    
    def _rule_based_evaluate(
        self,
        user_request: str,
        tool_result: str,
        criteria: List[str]
    ) -> FeedbackScore:
        """규칙 기반 평가 (LLM을 사용할 수 없을 때 fallback)"""
        
        try:
            # 기본적인 휴리스틱 평가
            scores = {}
            improvements = []
            strengths = []
            
            # 길이 기반 평가
            result_length = len(tool_result.strip())
            request_length = len(user_request.strip())
            
            if result_length < 10:
                length_score = 2
                improvements.append("결과가 너무 짧습니다")
            elif result_length < 50:
                length_score = 5  
                improvements.append("더 상세한 결과가 필요합니다")
            else:
                length_score = 8
                strengths.append("적절한 길이의 결과를 제공했습니다")
            
            # 에러 검사
            error_keywords = ["error", "failed", "exception", "오류", "실패"]
            has_error = any(keyword in tool_result.lower() for keyword in error_keywords)
            error_score = 2 if has_error else 8
            
            if has_error:
                improvements.append("실행 중 오류가 발생했습니다")
            else:
                strengths.append("오류 없이 실행되었습니다")
            
            # 키워드 매칭 점수
            request_words = set(user_request.lower().split())
            result_words = set(tool_result.lower().split())
            common_words = request_words.intersection(result_words)
            keyword_score = min(10, max(3, len(common_words) * 2))
            
            if len(common_words) > 2:
                strengths.append("요청과 관련된 내용을 포함하고 있습니다")
            else:
                improvements.append("요청과 더 관련성 높은 내용이 필요합니다")
            
            # 기준별 점수 계산
            base_score = (length_score + error_score + keyword_score) // 3
            
            for criterion in criteria:
                if criterion == "accuracy":
                    scores[criterion] = error_score
                elif criterion == "completeness":
                    scores[criterion] = length_score
                elif criterion == "relevance":
                    scores[criterion] = keyword_score
                else:
                    scores[criterion] = base_score
            
            overall_score = sum(scores.values()) // len(scores) if scores else base_score
            
            return FeedbackScore(
                overall_score=overall_score,
                criteria_scores=scores,
                reasoning=f"Rule-based evaluation: overall score {overall_score}/10",
                improvements=improvements,
                strengths=strengths
            )
            
        except Exception as e:
            logger.error(f"Rule-based evaluation failed: {str(e)}")
            return FeedbackScore(
                overall_score=5,
                criteria_scores={c: 5 for c in criteria},
                reasoning=f"Evaluation failed: {str(e)}",
                improvements=["평가 중 오류 발생"],
                strengths=["결과 생성 완료"]
            )

class IterativeImprover:
    """반복적 개선 도우미"""
    
    def __init__(self, evaluator: FeedbackEvaluator):
        self.evaluator = evaluator
        self.improvement_history: List[Dict] = []
    
    def suggest_improvements(
        self,
        user_request: str,
        current_result: str,
        feedback_score: FeedbackScore,
        context: Optional[str] = None
    ) -> str:
        """개선 제안 생성"""
        
        if feedback_score.overall_score >= 8:
            return current_result  # 이미 만족스러운 결과
        
        improvement_prompt = f"""
다음 개선사항들을 고려하여 더 나은 결과를 생성해주세요:

개선이 필요한 점들:
{chr(10).join(f"- {imp}" for imp in feedback_score.improvements)}

현재 점수가 낮은 기준들:
{chr(10).join(f"- {k}: {v}/10" for k, v in feedback_score.criteria_scores.items() if v < 7)}

이전 결과: {current_result}

위 피드백을 반영하여 사용자 요청을 더 잘 충족하는 결과를 생성해주세요.
"""
        
        return improvement_prompt
    
    def track_improvement(
        self,
        iteration: int,
        result: str,
        score: FeedbackScore
    ) -> None:
        """개선 과정 추적"""
        
        self.improvement_history.append({
            "iteration": iteration,
            "result": result,
            "score": score.overall_score,
            "criteria_scores": score.criteria_scores,
            "timestamp": __import__("time").time()
        })
    
    def get_best_result(self) -> Optional[Dict]:
        """가장 높은 점수의 결과 반환"""
        
        if not self.improvement_history:
            return None
        
        return max(self.improvement_history, key=lambda x: x["score"])
    
    def analyze_improvement_trend(self) -> Dict[str, Any]:
        """개선 추세 분석"""
        
        if len(self.improvement_history) < 2:
            return {"trend": "insufficient_data", "improvement": 0}
        
        scores = [entry["score"] for entry in self.improvement_history]
        first_score = scores[0]
        last_score = scores[-1]
        improvement = last_score - first_score
        
        if improvement > 2:
            trend = "improving"
        elif improvement > 0:
            trend = "slightly_improving"  
        elif improvement == 0:
            trend = "stable"
        else:
            trend = "declining"
        
        return {
            "trend": trend,
            "improvement": improvement,
            "total_iterations": len(self.improvement_history),
            "best_score": max(scores),
            "average_score": sum(scores) / len(scores)
        }

# 사전 정의된 평가 기준들
EVALUATION_TEMPLATES = {
    "general": [
        EvaluationCriteria.ACCURACY.value,
        EvaluationCriteria.COMPLETENESS.value,
        EvaluationCriteria.RELEVANCE.value,
        EvaluationCriteria.QUALITY.value
    ],
    "code_generation": [
        "functionality",
        "code_quality", 
        "best_practices",
        "completeness"
    ],
    "data_analysis": [
        "accuracy",
        "insight_quality",
        "data_coverage",
        "visualization"
    ],
    "text_generation": [
        "coherence",
        "relevance", 
        "style",
        "completeness"
    ]
}