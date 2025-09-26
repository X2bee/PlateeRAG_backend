import logging
from typing import Dict, Any
from editor.node_composer import Node

logger = logging.getLogger(__name__)

# TODO 실제 TOOL 결과에 대한 정보도 필요
class FeedbackCriteriaGeneratorNode(Node):
    categoryId = "xgen"
    functionId = "feedback"
    nodeId = "feedback/feedback_criteria_generator"
    nodeName = "Feedback Criteria Generator"
    description = "사용자 요청과 컨텍스트를 기반으로 피드백 평가 기준을 생성하는 노드입니다."
    tags = ["agent", "feedback", "criteria", "evaluation", "generator"]

    inputs = [
        {"id": "user_request", "name": "User Request", "type": "STR", "multi": False, "required": True, "stream": False}
    ]
    
    outputs = [
        {"id": "feedback_criteria", "name": "Feedback Criteria", "type": "FeedbackCrit", "required": True, "multi": False, "stream": False}
    ]
    
    parameters = [
        {
            "id": "task_type", 
            "name": "Task Type", 
            "type": "STR", 
            "value": "general", 
            "required": False,
            "options": [
                {"value": "general", "label": "General"},
                {"value": "coding", "label": "Coding"},
                {"value": "writing", "label": "Writing"},
                {"value": "analysis", "label": "Analysis"},
                {"value": "creative", "label": "Creative"},
                {"value": "technical", "label": "Technical"}
            ],
            "description": "작업 유형"
        },
        {
            "id": "quality_level", 
            "name": "Quality Level", 
            "type": "STR", 
            "value": "high", 
            "required": False,
            "options": [
                {"value": "basic", "label": "Basic"},
                {"value": "high", "label": "High"},
                {"value": "expert", "label": "Expert"}
            ],
            "description": "품질 요구 수준"
        },
        {
            "id": "domain", 
            "name": "Domain", 
            "type": "STR", 
            "value": "", 
            "required": False,
            "options": [
                {"value": "", "label": "None"},
                {"value": "programming", "label": "Programming"},
                {"value": "writing", "label": "Writing"},
                {"value": "design", "label": "Design"},
                {"value": "business", "label": "Business"}
            ],
            "description": "특화 도메인"
        },
        {
            "id": "custom_requirements", 
            "name": "Custom Requirements", 
            "type": "STR", 
            "value": "", 
            "expandable": True,
            "required": False,
            "description": "추가 커스텀 요구사항"
        },
        {
            "id": "criteria_type", 
            "name": "Criteria Type", 
            "type": "STR", 
            "value": "detailed", 
            "required": False,
            "options": [
                {"value": "simple", "label": "Simple"},
                {"value": "detailed", "label": "Detailed"},
                {"value": "domain_specific", "label": "Domain Specific"}
            ]
        },
        {
            "id": "include_scoring", 
            "name": "Include Scoring Guide", 
            "type": "BOOL", 
            "value": True, 
            "required": False,
            "description": "점수 체계 가이드 포함 여부"
        },
        {
            "id": "evaluation_aspects", 
            "name": "Evaluation Aspects", 
            "type": "STR", 
            "value": "accuracy,completeness,quality,usefulness", 
            "required": False,
            "options": [
                {"value": "accuracy,completeness,quality,usefulness", "label": "기본 (정확성, 완성도, 품질, 유용성)"},
                {"value": "accuracy,completeness,quality,clarity", "label": "문서/글쓰기 (정확성, 완성도, 품질, 명확성)"},
                {"value": "accuracy,efficiency,quality,feasibility", "label": "코딩/기술 (정확성, 효율성, 품질, 실행가능성)"},
                {"value": "creativity,quality,relevance,usefulness", "label": "창작/디자인 (창의성, 품질, 관련성, 유용성)"},
                {"value": "accuracy,completeness,clarity,relevance", "label": "분석/리서치 (정확성, 완성도, 명확성, 관련성)"},
                {"value": "accuracy,completeness,quality,usefulness,clarity,relevance,efficiency,creativity,feasibility", "label": "전체 (모든 측면 포함)"}
            ],
            "description": "평가할 측면들을 선택하세요"
        }
    ]

    def execute(
        self,
        user_request: str,
        task_type: str = "general",
        quality_level: str = "high",
        domain: str = "",
        custom_requirements: str = "",
        criteria_type: str = "detailed",
        include_scoring: bool = True,
        evaluation_aspects: str = "accuracy,completeness,quality,usefulness"
    ) -> Dict[str, Any]:
        """피드백 기준을 생성하여 반환"""
        
        try:
            # 평가 측면들을 리스트로 변환
            aspects = [aspect.strip() for aspect in evaluation_aspects.split(",") if aspect.strip()]
            
            # 기본 평가 기준 생성
            criteria = self._generate_base_criteria(user_request, task_type, quality_level, aspects)
            
            # 도메인 특화 기준 추가
            if domain and criteria_type in ["detailed", "domain_specific"]:
                criteria += self._add_domain_specific_criteria(domain)
            
            # 커스텀 요구사항 추가
            if custom_requirements:
                criteria += f"\n\n추가 요구사항:\n{custom_requirements}"

            criteria += self._add_feedback_loop_guidelines()

            # 점수 체계 가이드 추가
            if include_scoring:
                criteria += self._add_scoring_guide()
            
            # 최종 기준 정리
            if criteria_type == "simple":
                criteria = self._simplify_criteria(criteria)
            
            logger.info(f"Generated feedback criteria for request: {user_request[:50]}...")
            
            return {
                "feedback_criteria": criteria
            }
            
        except Exception as e:
            logger.error(f"Error generating feedback criteria: {str(e)}")
            # 기본 기준 반환
            default_criteria = """
사용자 요청을 얼마나 잘 충족했는지 평가해주세요:
- 정확성: 요청한 내용과 일치하는가?
- 완성도: 필요한 모든 정보가 포함되었는가?
- 품질: 결과의 품질이 높은가?
- 유용성: 사용자에게 도움이 되는가?

1-10점 척도로 평가하세요 (8점 이상이면 만족스러운 결과입니다).
"""
            return {
                "feedback_criteria": default_criteria
            }

    def _generate_base_criteria(self, user_request: str, task_type: str, quality_level: str, aspects: list) -> str:
        """기본 평가 기준 생성"""
        
        criteria_map = {
            "accuracy": "정확성: 요청한 내용과 얼마나 정확히 일치하는가?",
            "completeness": "완성도: 필요한 모든 정보와 요소가 포함되었는가?",
            "quality": "품질: 결과물의 전반적인 품질이 높은가?",
            "usefulness": "유용성: 사용자에게 실질적으로 도움이 되는가?",
            "clarity": "명확성: 이해하기 쉽고 명확하게 작성되었는가?",
            "relevance": "관련성: 사용자 요청과 직접적으로 관련이 있는가?",
            "efficiency": "효율성: 최적화되고 효율적인 해결책인가?",
            "creativity": "창의성: 독창적이고 창의적인 접근법을 사용했는가?",
            "feasibility": "실행가능성: 실제로 구현하거나 적용할 수 있는가?"
        }
        
        # 작업 유형별 특화 기준
        task_specific_criteria = {
            "coding": ["accuracy", "quality", "efficiency", "feasibility"],
            "writing": ["clarity", "completeness", "quality", "relevance"],
            "analysis": ["accuracy", "completeness", "usefulness", "clarity"],
            "creative": ["creativity", "quality", "usefulness", "relevance"],
            "technical": ["accuracy", "completeness", "efficiency", "feasibility"],
            "general": ["accuracy", "completeness", "quality", "usefulness"]
        }
        
        # 작업 유형에 따른 기본 측면 선택
        if task_type in task_specific_criteria:
            default_aspects = task_specific_criteria[task_type]
        else:
            default_aspects = task_specific_criteria["general"]
        
        # 사용자 지정 측면이 있으면 우선 사용, 없으면 기본 측면 사용
        final_aspects = aspects if aspects else default_aspects
        
        # 기준 생성
        criteria = f"사용자 요청: '{user_request}'\n\n"
        criteria += "위 요청에 대한 결과를 다음 기준으로 평가해주세요:\n\n"
        
        for i, aspect in enumerate(final_aspects, 1):
            if aspect in criteria_map:
                criteria += f"{i}. {criteria_map[aspect]}\n"
        
        # 품질 수준에 따른 추가 설명
        quality_descriptions = {
            "basic": "기본적인 요구사항을 충족하는지 확인",
            "high": "높은 품질과 완성도를 요구",
            "expert": "전문가 수준의 정확성과 깊이를 요구"
        }
        
        if quality_level in quality_descriptions:
            criteria += f"\n품질 기준: {quality_descriptions[quality_level]}\n"
        
        return criteria

    def _add_domain_specific_criteria(self, domain: str) -> str:
        """도메인 특화 기준 추가"""
        
        domain_criteria = {
            "programming": """
프로그래밍 특화 기준:
- 코드 품질: 가독성, 유지보수성, 모범 사례 준수
- 기능성: 요구된 기능이 올바르게 구현됨
- 성능: 효율적이고 최적화된 솔루션
- 보안: 보안 취약점이 없는 안전한 코드
""",
            "writing": """
글쓰기 특화 기준:
- 문체: 목적에 맞는 적절한 문체와 톤
- 구조: 논리적이고 체계적인 구성
- 문법: 올바른 문법과 맞춤법
- 가독성: 읽기 쉽고 이해하기 쉬운 표현
""",
            "design": """
디자인 특화 기준:
- 시각적 매력: 아름답고 매력적인 디자인
- 사용성: 사용자 친화적이고 직관적인 인터페이스
- 일관성: 통일된 디자인 언어와 스타일
- 접근성: 다양한 사용자가 접근 가능한 디자인
""",
            "business": """
비즈니스 특화 기준:
- 실용성: 비즈니스 목표와 요구사항에 부합
- 비용효율성: 투자 대비 효과가 높은 솔루션
- 확장성: 미래 성장에 대응 가능한 구조
- 위험성: 비즈니스 위험을 최소화하는 방향
"""
        }
        
        domain_lower = domain.lower()
        for key, value in domain_criteria.items():
            if key in domain_lower or domain_lower in key:
                return value
        
        return f"\n{domain} 도메인 특화 기준을 고려하여 평가하세요.\n"

    def _add_feedback_loop_guidelines(self) -> str:
        """피드백 루프 전용 운영 지침 추가"""

        return """

피드백 루프 운영 지침:
- TODO 단계별로 평가할 때는 해당 단계 목표 충족 여부만 검토하세요.
- 이전 단계 결과와의 일관성, 다음 단계로 전달 가능한지 여부를 확인하세요.
- 속도(불필요한 반복 없이 진행), 안정성(개선 사항 반영), 정확도(요구 충족)를 함께 고려하세요.
- 이미 완료된 작업을 중복 수행했거나 다음 단계 작업을 선행한 경우 감점 사유로 기록하세요.
"""

    def _add_scoring_guide(self) -> str:
        """점수 체계 가이드 추가"""
        
        return """

점수 체계 가이드:
1-2점: 매우 부족함 - 요구사항을 전혀 충족하지 못하거나 심각한 문제가 있음
3-4점: 부족함 - 일부 요구사항만 충족하고 많은 개선이 필요함
5-6점: 보통 - 기본 요구사항은 충족하나 품질이나 완성도가 아쉬움
7-8점: 좋음 - 대부분의 요구사항을 잘 충족하고 품질이 양호함
9-10점: 우수함 - 모든 요구사항을 완벽하게 충족하고 기대를 초과함

평가 시 각 기준별로 구체적인 이유를 제시하고, 개선이 필요한 부분과 잘된 부분을 명확히 구분해주세요.
"""

    def _simplify_criteria(self, criteria: str) -> str:
        """기준을 단순화"""
        
        lines = criteria.split('\n')
        simplified = []
        
        for line in lines:
            if line.strip() and not line.startswith('점수 체계') and '특화 기준' not in line:
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    simplified.append(line)
                elif line.startswith('사용자 요청'):
                    simplified.append(line)
                elif '평가해주세요' in line:
                    simplified.append(line)
        
        result = '\n'.join(simplified)
        result += "\n\n1-10점 척도로 평가하세요."
        
        return result
