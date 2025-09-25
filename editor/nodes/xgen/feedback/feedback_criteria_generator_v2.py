import logging
from typing import Dict, Any
from editor.node_composer import Node

logger = logging.getLogger(__name__)

# TODO 실제 TOOL 결과에 대한 정보도 필요
class FeedbackCriteriaGeneratorNode(Node):
    categoryId = "xgen"
    functionId = "feedback"
    nodeId = "feedback/feedback_criteria_generator_v2"
    nodeName = "Feedback Criteria Generator V2"
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
                {"value": "technical", "label": "Technical"},
                {"value": "research", "label": "Research"},
                {"value": "review", "label": "Review"},
                {"value": "product", "label": "Product/UX"}
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
                {"value": "expert", "label": "Expert"},
                {"value": "premium", "label": "Premium"}
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
                {"value": "business", "label": "Business"},
                {"value": "marketing", "label": "Marketing"},
                {"value": "education", "label": "Education"},
                {"value": "finance", "label": "Finance"}
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
                {"value": "domain_specific", "label": "Domain Specific"},
                {"value": "checklist", "label": "Checklist"},
                {"value": "rubric", "label": "Rubric"},
                {"value": "coaching", "label": "Coaching"}
            ],
            "description": "출력 형식을 선택하세요"
        },
        {
            "id": "evaluation_mode",
            "name": "Evaluation Mode",
            "type": "STR",
            "value": "balanced",
            "required": False,
            "options": [
                {"value": "balanced", "label": "균형 잡힌 평가"},
                {"value": "strict", "label": "엄격하고 보수적인 평가"},
                {"value": "supportive", "label": "지원/코칭 중심 평가"},
                {"value": "exploratory", "label": "탐색적/아이디어 중심"}
            ],
            "description": "평가 관점" 
        },
        {
            "id": "reviewer_persona",
            "name": "Reviewer Persona",
            "type": "STR",
            "value": "generalist",
            "required": False,
            "options": [
                {"value": "generalist", "label": "General Expert"},
                {"value": "qa_specialist", "label": "QA Specialist"},
                {"value": "mentor", "label": "Mentor/Coach"},
                {"value": "product_manager", "label": "Product Manager"},
                {"value": "end_user", "label": "End User"}
            ],
            "description": "평가자의 역할/관점"
        },
        {
            "id": "feedback_tone",
            "name": "Feedback Tone",
            "type": "STR",
            "value": "neutral",
            "required": False,
            "options": [
                {"value": "neutral", "label": "Neutral"},
                {"value": "encouraging", "label": "Encouraging"},
                {"value": "critical", "label": "Critical"},
                {"value": "empathetic", "label": "Empathetic"}
            ],
            "description": "피드백의 톤과 전달 방식"
        },
        {
            "id": "include_summary",
            "name": "Include Summary",
            "type": "BOOL",
            "value": True,
            "required": False,
            "description": "요약 섹션 포함 여부"
        },
        {
            "id": "summary_style",
            "name": "Summary Style",
            "type": "STR",
            "value": "bullet",
            "required": False,
            "options": [
                {"value": "bullet", "label": "주요 포인트 (불릿)"},
                {"value": "paragraph", "label": "단락 요약"},
                {"value": "key_takeaways", "label": "핵심 시사점"}
            ],
            "description": "요약 형식",
            "dependency": "include_summary"
        },
        {
            "id": "include_improvement_tips",
            "name": "Include Improvement Tips",
            "type": "BOOL",
            "value": True,
            "required": False,
            "description": "개선 가이드 포함 여부"
        },
        {
            "id": "improvement_focus",
            "name": "Improvement Focus",
            "type": "STR",
            "value": "actionable",
            "required": False,
            "options": [
                {"value": "actionable", "label": "즉시 실행 가능한 조치"},
                {"value": "root_cause", "label": "근본 원인 분석"},
                {"value": "user_experience", "label": "사용자 경험 개선"},
                {"value": "risk_mitigation", "label": "위험 완화"}
            ],
            "description": "개선안 초점",
            "dependency": "include_improvement_tips"
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
            "id": "scoring_style",
            "name": "Scoring Style",
            "type": "STR",
            "value": "scale_1_10",
            "required": False,
            "options": [
                {"value": "scale_1_10", "label": "1-10 점수"},
                {"value": "scale_1_5", "label": "1-5 점수"},
                {"value": "pass_fail", "label": "Pass / Fail"},
                {"value": "rubric_levels", "label": "Rubric (Excellent~Poor)"}
            ],
            "description": "점수 표현 방식",
            "dependency": "include_scoring"
        },
        {
            "id": "include_weighting",
            "name": "Include Weighting Guide",
            "type": "BOOL",
            "value": False,
            "required": False,
            "description": "각 평가 항목별 가중치 제시",
            "dependency": "include_scoring"
        },
        {
            "id": "weighting_strategy",
            "name": "Weighting Strategy",
            "type": "STR",
            "value": "balanced",
            "required": False,
            "options": [
                {"value": "balanced", "label": "균형 배분"},
                {"value": "accuracy_focus", "label": "정확성 중점"},
                {"value": "quality_focus", "label": "품질/완성도 중점"},
                {"value": "custom", "label": "사용자 지정"}
            ],
            "description": "가중치 전략",
            "dependency": "include_weighting"
        },
        {
            "id": "custom_weights",
            "name": "Custom Weights",
            "type": "STR",
            "value": "",
            "required": False,
            "expandable": True,
            "description": "예: accuracy:40,quality:30,clarity:30",
            "dependency": "include_weighting"
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
                {"value": "accuracy,impact,risk,clarity", "label": "리뷰/감리 (정확성, 영향도, 위험, 명확성)"},
                {"value": "usefulness,user_experience,accuracy,feasibility", "label": "제품/UX (유용성, UX, 정확성, 실행가능성)"},
                {"value": "accuracy,completeness,quality,usefulness,clarity,relevance,efficiency,creativity,feasibility", "label": "전체 (모든 측면 포함)"}
            ],
            "description": "평가할 측면들을 선택하세요"
        },
        {
            "id": "enable_custom_aspects",
            "name": "Enable Custom Aspects",
            "type": "BOOL",
            "value": False,
            "required": False,
            "description": "직접 평가 항목 정의"
        },
        {
            "id": "custom_aspects",
            "name": "Custom Aspects",
            "type": "STR",
            "value": "",
            "required": False,
            "expandable": True,
            "description": "콤마로 구분된 사용자 지정 평가 항목",
            "dependency": "enable_custom_aspects"
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
        evaluation_mode: str = "balanced",
        reviewer_persona: str = "generalist",
        feedback_tone: str = "neutral",
        include_summary: bool = True,
        summary_style: str = "bullet",
        include_improvement_tips: bool = True,
        improvement_focus: str = "actionable",
        include_scoring: bool = True,
        scoring_style: str = "scale_1_10",
        include_weighting: bool = False,
        weighting_strategy: str = "balanced",
        custom_weights: str = "",
        evaluation_aspects: str = "accuracy,completeness,quality,usefulness",
        enable_custom_aspects: bool = False,
        custom_aspects: str = ""
    ) -> Dict[str, Any]:
        """피드백 기준을 생성하여 반환"""
        
        try:
            # 평가 측면들을 리스트로 변환
            if enable_custom_aspects and custom_aspects.strip():
                aspects = [aspect.strip() for aspect in custom_aspects.split(",") if aspect.strip()]
            else:
                aspects = [aspect.strip() for aspect in evaluation_aspects.split(",") if aspect.strip()]

            # 기본 평가 기준 생성
            criteria = self._generate_base_criteria(
                user_request,
                task_type,
                quality_level,
                aspects,
                evaluation_mode,
                reviewer_persona,
                feedback_tone,
            )

            # 도메인 특화 기준 추가
            if domain and criteria_type in [
                "detailed",
                "domain_specific",
                "rubric",
                "checklist",
                "coaching",
            ]:
                criteria += self._add_domain_specific_criteria(domain)

            # 커스텀 요구사항 추가
            if custom_requirements:
                criteria += f"\n\n추가 요구사항:\n{custom_requirements}"

            if include_weighting and aspects:
                criteria += self._add_weighting_guidelines(aspects, weighting_strategy, custom_weights)

            if include_improvement_tips:
                criteria += self._add_improvement_section(improvement_focus, feedback_tone)

            if include_summary:
                criteria += self._add_summary_section(summary_style)

            criteria += self._add_feedback_loop_guidelines(evaluation_mode, feedback_tone)

            # 점수 체계 가이드 추가
            if include_scoring:
                criteria += self._add_scoring_guide(scoring_style)

            # 최종 기준 정리
            criteria = self._apply_criteria_type(criteria, criteria_type, aspects, scoring_style)

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

    def _get_criteria_map(self) -> Dict[str, str]:
        """평가 항목과 설명 매핑"""

        return {
            "accuracy": "정확성: 요청한 내용과 얼마나 정확히 일치하는가?",
            "completeness": "완성도: 필요한 모든 정보와 요소가 포함되었는가?",
            "quality": "품질: 결과물의 전반적인 품질이 높은가?",
            "usefulness": "유용성: 사용자에게 실질적으로 도움이 되는가?",
            "clarity": "명확성: 이해하기 쉽고 명확하게 작성되었는가?",
            "relevance": "관련성: 사용자 요청과 직접적으로 관련이 있는가?",
            "efficiency": "효율성: 최적화되고 효율적인 해결책인가?",
            "creativity": "창의성: 독창적이고 창의적인 접근법을 사용했는가?",
            "feasibility": "실행가능성: 실제로 구현하거나 적용할 수 있는가?",
            "impact": "영향도: 결과물이 사용자 또는 비즈니스에 미치는 긍정적 영향이 충분한가?",
            "risk": "위험 관리: 잠재적 위험 요소를 식별하고 완화했는가?",
            "user_experience": "사용자 경험: 흐름, 접근성, 만족도 측면에서 우수한가?",
            "maintainability": "유지보수성: 향후 관리와 확장이 용이한 구조인가?",
            "evidence": "근거 제시: 주장이나 결론을 뒷받침하는 근거가 충분한가?",
        }

    def _generate_base_criteria(
        self,
        user_request: str,
        task_type: str,
        quality_level: str,
        aspects: list,
        evaluation_mode: str,
        reviewer_persona: str,
        feedback_tone: str,
    ) -> str:
        """기본 평가 기준 생성"""

        criteria_map = self._get_criteria_map()

        # 작업 유형별 특화 기준
        task_specific_criteria = {
            "coding": ["accuracy", "quality", "efficiency", "feasibility"],
            "writing": ["clarity", "completeness", "quality", "relevance"],
            "analysis": ["accuracy", "completeness", "usefulness", "clarity"],
            "creative": ["creativity", "quality", "usefulness", "relevance"],
            "technical": ["accuracy", "completeness", "efficiency", "feasibility"],
            "research": ["accuracy", "completeness", "clarity", "relevance"],
            "review": ["accuracy", "impact", "risk", "clarity"],
            "product": ["usefulness", "user_experience", "feasibility", "impact"],
            "general": ["accuracy", "completeness", "quality", "usefulness"],
        }

        # 작업 유형에 따른 기본 측면 선택
        default_aspects = task_specific_criteria.get(task_type, task_specific_criteria["general"])

        # 사용자 지정 측면이 있으면 우선 사용, 없으면 기본 측면 사용
        final_aspects = aspects if aspects else default_aspects

        # 기준 생성
        criteria = f"사용자 요청: '{user_request}'\n\n"
        criteria += "위 요청에 대한 결과를 다음 기준으로 평가해주세요:\n\n"

        for i, aspect in enumerate(final_aspects, 1):
            if aspect in criteria_map:
                criteria += f"{i}. {criteria_map[aspect]}\n"
            else:
                criteria += f"{i}. {aspect}: 작업 목표와의 적합성을 검토하세요.\n"

        # 품질 수준에 따른 추가 설명
        quality_descriptions = {
            "basic": "기본적인 요구사항을 충족하는지 확인",
            "high": "높은 품질과 완성도를 요구",
            "expert": "전문가 수준의 정확성과 깊이를 요구",
            "premium": "전문가 품질과 비즈니스 영향까지 고려한 최상급 결과를 요구",
        }

        if quality_level in quality_descriptions:
            criteria += f"\n품질 기준: {quality_descriptions[quality_level]}\n"

        criteria += f"평가 모드: {self._describe_evaluation_mode(evaluation_mode)}\n"
        criteria += f"검토자 관점: {self._describe_reviewer_persona(reviewer_persona)}\n"
        criteria += f"피드백 톤: {self._describe_feedback_tone(feedback_tone)}\n"

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
""",
            "marketing": """
마케팅 특화 기준:
- 메시지 적합성: 타깃 고객과 브랜드 톤에 부합하는가?
- 전환 가능성: 행동을 유도하는 명확한 CTA와 가치 제안이 있는가?
- 데이터 활용: 근거 자료나 지표가 뒷받침되는가?
- 측정 계획: 성과 측정 지표와 추적 방법이 정의되어 있는가?
""",
            "education": """
교육 특화 기준:
- 학습 목표 정렬: 콘텐츠가 학습 목표를 달성하도록 구성되었는가?
- 난이도 적합성: 대상 학습자 수준과 인지 부담을 고려했는가?
- 상호작용성: 학습 참여를 유도할 활동이나 질문이 있는가?
- 평가 방법: 학습 성취도를 확인할 수 있는 평가가 준비되었는가?
""",
            "finance": """
금융 특화 기준:
- 정확성: 수치, 계산, 규제 준수 여부가 정확한가?
- 리스크 분석: 잠재적 위험과 대비 방안이 포함되었는가?
- 수익성/효율성: 재무적 영향이 명확히 분석되었는가?
- 투명성: 가정과 근거가 명확히 제시되었는가?
"""
        }
        
        domain_lower = domain.lower()
        for key, value in domain_criteria.items():
            if key in domain_lower or domain_lower in key:
                return value
        
        return f"\n{domain} 도메인 특화 기준을 고려하여 평가하세요.\n"

    def _describe_evaluation_mode(self, evaluation_mode: str) -> str:
        mode_map = {
            "balanced": "강점과 개선점을 균형 있게 다루고, 명확한 근거를 중심으로 판단합니다.",
            "strict": "기준을 엄격히 적용하고 사소한 오류도 기록하며, 품질을 최우선으로 평가합니다.",
            "supportive": "코칭 관점에서 성장 가능성과 개선 방향을 먼저 제시합니다.",
            "exploratory": "새로운 아이디어와 실험을 장려하며, 통찰과 가능성을 중심으로 평가합니다.",
        }
        return mode_map.get(evaluation_mode, mode_map["balanced"])

    def _describe_reviewer_persona(self, reviewer_persona: str) -> str:
        persona_map = {
            "generalist": "종합 전문가 시각에서 전반적인 완성도와 일관성을 검토합니다.",
            "qa_specialist": "리스크와 결함을 우선 확인하는 QA 검토자의 시각을 유지합니다.",
            "mentor": "멘토로서 성장 기회를 발굴하고 구체적인 개선 제안을 제시합니다.",
            "product_manager": "사용자 가치와 비즈니스 영향, 우선순위를 중점적으로 검토합니다.",
            "end_user": "최종 사용자로서 실사용 관점과 경험적 만족도를 확인합니다.",
        }
        return persona_map.get(reviewer_persona, persona_map["generalist"])

    def _describe_feedback_tone(self, feedback_tone: str) -> str:
        tone_map = {
            "neutral": "중립적이고 객관적인 어조로 사실과 근거를 중심으로 전달합니다.",
            "encouraging": "긍정적인 표현을 활용해 동기를 부여하면서 개선점을 안내합니다.",
            "critical": "명확하고 단호하게 문제점을 짚고 시급한 조치를 제안합니다.",
            "empathetic": "이해와 공감을 전제로 개선 방향을 부드럽게 제안합니다.",
        }
        return tone_map.get(feedback_tone, tone_map["neutral"])

    def _add_feedback_loop_guidelines(self, evaluation_mode: str, feedback_tone: str) -> str:
        """피드백 루프 전용 운영 지침 추가"""

        mode_tip_map = {
            "strict": "평가 기준을 엄격히 유지하되, 재검토가 필요한 부분은 명시하세요.",
            "supportive": "개선 가능성과 학습 포인트를 먼저 언급하고 구체적 실행 단계를 안내하세요.",
            "exploratory": "실험적 시도를 존중하면서도 목표와의 연결성을 확인하세요.",
        }
        tone_tip_map = {
            "encouraging": "격려 표현을 활용해 다음 단계에 대한 자신감을 심어주세요.",
            "critical": "문제의 심각도를 명확히 표현하되 근거와 대안을 함께 제시하세요.",
            "empathetic": "문제를 지적할 때도 공감 표현을 곁들여 수용성을 높이세요.",
        }

        additional_tips = []
        if evaluation_mode in mode_tip_map:
            additional_tips.append(f"- {mode_tip_map[evaluation_mode]}")
        if feedback_tone in tone_tip_map:
            additional_tips.append(f"- {tone_tip_map[feedback_tone]}")

        tips_text = "\n".join(additional_tips)

        return f"""

피드백 루프 운영 지침:
- TODO 단계별로 평가할 때는 해당 단계 목표 충족 여부만 검토하세요.
- 이전 단계 결과와의 일관성, 다음 단계로 전달 가능한지 여부를 확인하세요.
- 속도(불필요한 반복 없이 진행), 안정성(개선 사항 반영), 정확도(요구 충족)를 함께 고려하세요.
- 이미 완료된 작업을 중복 수행했거나 다음 단계 작업을 선행한 경우 감점 사유로 기록하세요.
{tips_text if tips_text else ''}
"""

    def _add_scoring_guide(self, scoring_style: str) -> str:
        """점수 체계 가이드 추가"""

        if scoring_style == "scale_1_5":
            guide = """

점수 체계 가이드 (1-5 척도):
1점: 매우 부족함 - 핵심 요구를 충족하지 못함
2점: 부족함 - 부분적으로 충족했으나 주요 개선이 필요함
3점: 보통 - 기본 요구를 충족하지만 품질 향상이 필요함
4점: 좋음 - 대부분의 항목을 충족하며 개선 포인트가 명확함
5점: 우수함 - 기대를 초과하며 명확한 강점이 돋보임
"""
        elif scoring_style == "pass_fail":
            guide = """

점수 체계 가이드 (Pass / Fail):
Pass: 요구사항을 충족하며 사용 준비가 완료된 상태
Fail: 핵심 요구사항을 충족하지 못해 재작업이 필요한 상태

판정 근거와 재작업이 필요한 구체 항목을 함께 기록하세요.
"""
        elif scoring_style == "rubric_levels":
            guide = """

점수 체계 가이드 (루브릭):
Excellent: 기대를 초과하는 매우 우수한 결과
Good: 대부분의 요구를 충족하며 약간의 개선만 필요
Fair: 핵심 요구는 충족하지만 품질/완성도 개선 필요
Poor: 요구 충족도가 낮거나 근본적 개선이 필요함

각 평가 항목마다 등급 판정 근거를 간단히 설명하세요.
"""
        else:
            guide = """

점수 체계 가이드 (1-10 척도):
1-2점: 매우 부족함 - 요구사항을 전혀 충족하지 못하거나 심각한 문제가 있음
3-4점: 부족함 - 일부 요구사항만 충족하고 많은 개선이 필요함
5-6점: 보통 - 기본 요구사항은 충족하나 품질이나 완성도가 아쉬움
7-8점: 좋음 - 대부분의 요구사항을 잘 충족하고 품질이 양호함
9-10점: 우수함 - 모든 요구사항을 완벽하게 충족하고 기대를 초과함
"""

        return guide + "\n평가 시 각 기준별로 구체적인 이유를 제시하고, 개선이 필요한 부분과 잘된 부분을 명확히 구분해주세요.\n"

    def _add_summary_section(self, summary_style: str) -> str:
        style_map = {
            "bullet": "3-5개의 핵심 포인트를 불릿으로 정리하고, 각 포인트에 근거를 덧붙이세요.",
            "paragraph": "2-3문장으로 주요 성과와 부족한 점을 균형 있게 서술하세요.",
            "key_takeaways": "핵심 시사점/결정 사항을 위주로 요약하고 다음 단계 권장사항을 포함하세요.",
        }
        guidance = style_map.get(summary_style, style_map["bullet"])
        return f"\n요약 작성 지침:\n- {guidance}\n"

    def _add_improvement_section(self, improvement_focus: str, feedback_tone: str) -> str:
        focus_map = {
            "actionable": "즉시 실행 가능한 2-3개의 구체적인 개선 조치를 제시하세요.",
            "root_cause": "문제의 근본 원인을 추적하고 이를 해결할 구조적 개선안을 제안하세요.",
            "user_experience": "사용자 흐름과 감정선을 고려해 UX 향상 방안을 제시하세요.",
            "risk_mitigation": "주요 리스크와 그 영향도를 정리하고 완화 전략을 제안하세요.",
        }
        tone_hint = {
            "encouraging": "강점을 언급한 뒤 개선안을 제시해 수용성을 높이세요.",
            "critical": "우선순위가 높은 문제부터 단호하게 제시하되 근거를 명확히 밝히세요.",
            "empathetic": "현재 노력과 상황을 인정하면서 개선안을 제시하세요.",
        }.get(feedback_tone)

        tips = [focus_map.get(improvement_focus, focus_map["actionable"])]
        if tone_hint:
            tips.append(tone_hint)

        tips_text = "\n- ".join(tips)
        return f"\n개선 가이드라인:\n- {tips_text}\n"

    def _add_weighting_guidelines(self, aspects: list, weighting_strategy: str, custom_weights: str) -> str:
        weights = self._create_weight_map(aspects, weighting_strategy, custom_weights)
        if not weights:
            return ""

        lines = ["\n가중치 가이드:"]
        for aspect, weight in weights.items():
            label = self._get_aspect_label(aspect)
            lines.append(self._format_weight_line(label, weight))

        lines.append("가중치는 판단 근거와 함께 기록하고, 필요 시 다음 반복에서 조정하세요.")
        return "\n".join(lines) + "\n"

    def _create_weight_map(self, aspects: list, weighting_strategy: str, custom_weights: str) -> Dict[str, float]:
        if not aspects:
            return {}

        normalized_aspects = [aspect.strip() for aspect in aspects if aspect.strip()]
        if not normalized_aspects:
            return {}

        if weighting_strategy == "custom" and custom_weights:
            weights = {}
            for item in custom_weights.split(","):
                if ":" in item:
                    key, value = item.split(":", 1)
                    key = key.strip()
                    try:
                        weights[key] = float(value.strip())
                    except ValueError:
                        continue
            total = sum(weights.values())
            if total > 0:
                return {k: round(v / total * 100, 2) for k, v in weights.items() if k}
            return weights

        base_value = round(100 / len(normalized_aspects), 2)
        weights = {aspect: base_value for aspect in normalized_aspects}

        focus_map = {
            "accuracy_focus": "accuracy",
            "quality_focus": "quality",
        }

        focus_aspect = focus_map.get(weighting_strategy)
        if focus_aspect and focus_aspect in weights:
            focus_bonus = 20
            remaining = max(0, 100 - focus_bonus)
            others = len(normalized_aspects) - 1 or 1
            distributed = round(remaining / others, 2)
            for aspect in normalized_aspects:
                if aspect == focus_aspect:
                    weights[aspect] = focus_bonus
                else:
                    weights[aspect] = distributed
        return weights

    def _format_weight_line(self, label: str, weight: float) -> str:
        return f"- {label}: 약 {round(weight, 2)}% 비중으로 평가"

    def _get_aspect_label(self, aspect: str) -> str:
        criteria_map = self._get_criteria_map()
        description = criteria_map.get(aspect, aspect)
        return description.split(":", 1)[0]

    def _apply_criteria_type(self, criteria: str, criteria_type: str, aspects: list, scoring_style: str) -> str:
        if criteria_type == "simple":
            return self._simplify_criteria(criteria, scoring_style)
        if criteria_type == "checklist":
            return criteria + self._convert_to_checklist(aspects)
        if criteria_type == "rubric":
            return criteria + self._build_rubric_template(aspects, scoring_style)
        if criteria_type == "coaching":
            return criteria + self._build_coaching_prompts(aspects)
        return criteria

    def _convert_to_checklist(self, aspects: list) -> str:
        if not aspects:
            return ""
        lines = ["\n체크리스트:"]
        for aspect in aspects:
            label = self._get_aspect_label(aspect)
            lines.append(f"- [ ] {label}: 기준 충족 여부와 근거를 기록")
        return "\n".join(lines) + "\n"

    def _build_rubric_template(self, aspects: list, scoring_style: str) -> str:
        if not aspects:
            return ""
        headers = "| 항목 | 최고 수준 | 양호 수준 | 개선 필요 |"
        separator = "| --- | --- | --- | --- |"
        rows = []
        for aspect in aspects:
            label = self._get_aspect_label(aspect)
            rows.append(f"| {label} | 기대를 초과 | 대부분 충족 | 추가 개선 필요 |")
        rubric_intro = "\n루브릭 템플릿:" if scoring_style == "rubric_levels" else "\n평가 루브릭 제안:" 
        return "\n".join([rubric_intro, headers, separator, *rows, ""])

    def _build_coaching_prompts(self, aspects: list) -> str:
        prompts = ["\n코칭형 피드백 가이드:", "- 강점 -> 개선 -> 다음 행동 순서로 전달", "- 개선안은 실행 주체, 기한, 기대 효과를 포함"]
        if aspects:
            aspect_lines = ", ".join(self._get_aspect_label(aspect) for aspect in aspects)
            prompts.append(f"- 각 항목({aspect_lines})별 칭찬과 구체적 개선안을 1개 이상 제안")
        return "\n".join(prompts) + "\n"

    def _simplify_criteria(self, criteria: str, scoring_style: str) -> str:
        """기준을 단순화"""

        lines = criteria.split('\n')
        simplified = []
        number_prefixes = tuple(f"{i}." for i in range(1, 10))

        for line in lines:
            if line.strip() and not line.startswith('점수 체계') and '특화 기준' not in line:
                if line.startswith(number_prefixes):
                    simplified.append(line)
                elif line.startswith('사용자 요청'):
                    simplified.append(line)
                elif '평가해주세요' in line:
                    simplified.append(line)

        result = '\n'.join(simplified)

        scoring_hint = {
            "scale_1_5": "1-5점 척도로 평가하세요.",
            "pass_fail": "Pass / Fail로 판정하세요.",
            "rubric_levels": "루브릭 등급으로 평가를 마무리하세요.",
        }.get(scoring_style, "1-10점 척도로 평가하세요.")

        result += f"\n\n{scoring_hint}"

        return result
