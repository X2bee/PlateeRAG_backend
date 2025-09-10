import logging
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedTreeSearchAlgorithm:
    """
    고도화된 트리 서치 알고리즘 클래스
    - Hierarchical Directory Analysis
    - Multi-level Scoring System
    - Diversity-aware Selection
    - Adaptive Path Weighting
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1, diversity_threshold: float = 0.7):
        """
        alpha: 원본 스코어 가중치
        beta: 트리 구조 가중치
        gamma: 다양성 가중치
        diversity_threshold: 다양성 임계값
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.diversity_threshold = diversity_threshold

    def build_directory_tree(self, results: List[Dict]) -> Dict:
        """디렉토리 트리 구조 생성"""
        tree = defaultdict(lambda: defaultdict(list))

        for item in results:
            path = item.get("directory_full_path", "")
            if path:
                # 경로를 계층적으로 분해
                parts = [p for p in path.split('/') if p]

                # 각 레벨별로 문서 저장
                for i in range(len(parts)):
                    level_path = '/'.join(parts[:i+1])
                    tree[i][level_path].append(item)

        return dict(tree)

    def calculate_path_weights(self, tree: Dict, results: List[Dict]) -> Dict[str, float]:
        """경로별 가중치 계산 (계층적 분석)"""
        path_weights = {}
        total_docs = len(results)

        # 각 레벨별 분석
        for level, level_data in tree.items():
            level_weight = 1.0 / (level + 1)  # 깊은 레벨일수록 가중치 감소

            for path, docs in level_data.items():
                doc_count = len(docs)
                avg_score = np.mean([doc.get("score", 0.0) for doc in docs])

                # 문서 밀도와 품질을 고려한 가중치
                density_score = doc_count / total_docs
                quality_score = avg_score

                # 경로 가중치 = 레벨 가중치 × 밀도 × 품질
                path_weights[path] = level_weight * density_score * quality_score

        return path_weights

    def calculate_diversity_penalty(self, selected_items: List[Dict], candidate: Dict) -> float:
        """다양성 패널티 계산"""
        if not selected_items:
            return 0.0

        candidate_path = candidate.get("directory_full_path", "")

        # 이미 선택된 항목들과의 경로 유사도 계산
        similarities = []
        for item in selected_items:
            item_path = item.get("directory_full_path", "")
            similarity = self._calculate_path_similarity(candidate_path, item_path)
            similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        # 유사도가 높을수록 패널티 증가
        return avg_similarity if avg_similarity > self.diversity_threshold else 0.0

    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        """두 경로 간 유사도 계산"""
        if not path1 or not path2:
            return 0.0

        parts1 = set(path1.split('/'))
        parts2 = set(path2.split('/'))

        if not parts1 or not parts2:
            return 0.0

        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))

        return intersection / union if union > 0 else 0.0

    def monte_carlo_tree_search(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Monte Carlo Tree Search 기반 최적화"""
        if len(results) <= top_k:
            return results

        # 1. 트리 구조 생성
        tree = self.build_directory_tree(results)
        logger.info("디렉토리 트리 생성 완료: %d 레벨", len(tree))

        # 2. 경로별 가중치 계산
        path_weights = self.calculate_path_weights(tree, results)
        logger.info("경로별 가중치 계산 완료: %d 경로", len(path_weights))

        # 3. 각 결과에 대해 종합 스코어 계산
        for item in results:
            original_score = item.get("score", 0.0)
            path = item.get("directory_full_path", "")

            # 트리 구조 스코어 (모든 상위 경로의 가중치 합)
            tree_score = 0.0
            if path:
                parts = [p for p in path.split('/') if p]
                for i in range(len(parts)):
                    partial_path = '/'.join(parts[:i+1])
                    tree_score += path_weights.get(partial_path, 0.0)

            # 정규화
            tree_score = tree_score / len(parts) if path and parts else 0.0

            # 임시 종합 스코어 (다양성 제외)
            item["tree_score"] = tree_score
            item["temp_composite_score"] = self.alpha * original_score + self.beta * tree_score

        # 4. 다양성을 고려한 Greedy Selection
        selected_items = []
        remaining_items = sorted(results, key=lambda x: x["temp_composite_score"], reverse=True)

        for _ in range(min(top_k, len(remaining_items))):
            best_item = None
            best_score = -1.0
            best_index = -1

            for i, candidate in enumerate(remaining_items):
                # 다양성 패널티 계산
                diversity_penalty = self.calculate_diversity_penalty(selected_items, candidate)

                # 최종 스코어 계산
                final_score = (candidate["temp_composite_score"] -
                             self.gamma * diversity_penalty)

                if final_score > best_score:
                    best_score = final_score
                    best_item = candidate
                    best_index = i

            if best_item:
                best_item["final_score"] = best_score
                selected_items.append(best_item)
                remaining_items.pop(best_index)

                logger.debug("선택된 문서: %s, 최종스코어: %.4f, 경로: %s",
                           best_item.get('file_name', 'Unknown'),
                           best_score,
                           best_item.get('directory_full_path', 'Unknown'))

        return selected_items

    def apply_quality_filtering(self, results: List[Dict], min_quality_score: float = 0.15) -> List[Dict]:
        """품질 기준에 따른 결과 필터링"""
        if not results:
            return results

        # 최종 스코어 기준으로 필터링
        filtered_results = []
        for item in results:
            final_score = item.get("final_score", item.get("score", 0.0))
            if final_score >= min_quality_score:
                filtered_results.append(item)

        logger.info("품질 필터링: %d개 → %d개 (임계값: %.2f)",
                   len(results), len(filtered_results), min_quality_score)

        return filtered_results

    def dynamic_top_k_selection(self, results: List[Dict], requested_top_k: int,
                               min_quality_score: float = 0.15) -> tuple[List[Dict], Dict[str, Any]]:
        """동적 Top-K 선택 (품질 기준 적용)"""
        if not results:
            return [], {"strategy": "no_results", "message": "검색 결과가 없습니다."}

        # 품질 필터링 적용
        quality_filtered = self.apply_quality_filtering(results, min_quality_score)

        # 전략 결정
        if len(quality_filtered) == 0:
            # 품질 기준을 낮춰서 재시도
            relaxed_threshold = min_quality_score * 0.6
            quality_filtered = self.apply_quality_filtering(results, relaxed_threshold)

            if len(quality_filtered) == 0:
                return results[:1], {  # 최소 1개는 반환
                    "strategy": "emergency_fallback",
                    "message": f"품질 기준({min_quality_score:.2f})을 만족하는 결과가 없어 최상위 1개만 반환합니다.",
                    "warning": "결과의 품질이 매우 낮을 수 있습니다."
                }
            else:
                return quality_filtered[:requested_top_k], {
                    "strategy": "relaxed_filtering",
                    "message": f"품질 기준을 {relaxed_threshold:.2f}로 완화하여 {len(quality_filtered)}개 결과를 반환합니다."
                }

        elif len(quality_filtered) < requested_top_k:
            return quality_filtered, {
                "strategy": "partial_results",
                "message": f"요청한 {requested_top_k}개 중 품질 기준을 만족하는 {len(quality_filtered)}개만 반환합니다.",
                "actual_count": len(quality_filtered),
                "requested_count": requested_top_k
            }
        else:
            return quality_filtered[:requested_top_k], {
                "strategy": "full_results",
                "message": f"품질 기준을 만족하는 {requested_top_k}개 결과를 반환합니다."
            }

    def analyze_selection_quality(self, selected_items: List[Dict]) -> Dict[str, Any]:
        """선택 결과 품질 분석"""
        if not selected_items:
            return {
                "quality_grade": "F",
                "recommendations": ["검색 결과가 없습니다. 검색 조건을 완화해보세요."],
                "diversity_score": 0.0,
                "unique_paths": 0,
                "avg_original_score": 0.0,
                "avg_final_score": 0.0,
                "path_distribution": {},
                "result_adequacy": "insufficient"
            }

        paths = [item.get("directory_full_path", "") for item in selected_items if item.get("directory_full_path")]
        unique_paths = set(paths)

        diversity_score = len(unique_paths) / len(selected_items) if selected_items else 0.0
        avg_score = np.mean([item.get("score", 0.0) for item in selected_items])
        avg_final_score = np.mean([item.get("final_score", 0.0) for item in selected_items])

        # 점수 분산 계산 (일관성 측정)
        score_variance = np.var([item.get("final_score", 0.0) for item in selected_items])
        consistency_score = 1.0 / (1.0 + score_variance)  # 분산이 낮을수록 일관성 높음

        path_distribution = defaultdict(int)
        for path in paths:
            path_distribution[path] += 1

        # 품질 등급 계산 (A, B, C, D, F)
        quality_grade = self._calculate_quality_grade(diversity_score, avg_final_score, consistency_score)

        # 개선 권장사항 생성
        recommendations = self._generate_recommendations(diversity_score, avg_final_score, consistency_score, len(selected_items))

        # 경로 집중도 분석 (특정 경로에 너무 집중되었는지)
        max_concentration = max(path_distribution.values()) if path_distribution else 0
        concentration_ratio = max_concentration / len(selected_items) if selected_items else 0

        # 결과 충분성 평가
        result_adequacy = self._assess_result_adequacy(selected_items, avg_final_score)

        return {
            "quality_grade": quality_grade,
            "recommendations": recommendations,
            "diversity_score": diversity_score,
            "consistency_score": consistency_score,
            "concentration_ratio": concentration_ratio,
            "unique_paths": len(unique_paths),
            "avg_original_score": avg_score,
            "avg_final_score": avg_final_score,
            "score_improvement": avg_final_score - avg_score,  # 알고리즘의 개선 효과
            "path_distribution": dict(path_distribution),
            "result_adequacy": result_adequacy
        }

    def _assess_result_adequacy(self, selected_items: List[Dict], avg_final_score: float) -> str:
        """결과 충분성 평가 (0.3=보통, 0.5+=높음 기준)"""
        if len(selected_items) == 0:
            return "none"
        elif len(selected_items) == 1:
            return "minimal" if avg_final_score >= 0.3 else "insufficient"
        elif len(selected_items) <= 2:
            return "limited" if avg_final_score >= 0.25 else "insufficient"
        elif avg_final_score < 0.2:
            return "poor_quality"
        else:
            return "adequate"

    def _calculate_quality_grade(self, diversity_score: float, avg_final_score: float, consistency_score: float) -> str:
        """품질 등급 계산"""
        # 종합 점수 계산 (0~1 범위)
        composite_score = (diversity_score * 0.4 + avg_final_score * 0.4 + consistency_score * 0.2)

        if composite_score >= 0.6:
            return "A"
        elif composite_score >= 0.5:
            return "B"
        elif composite_score >= 0.4:
            return "C"
        elif composite_score >= 0.3:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self, diversity_score: float, avg_final_score: float,
                                consistency_score: float, total_items: int) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        # 결과 수량 관련 권장사항
        if total_items == 0:
            recommendations.append("검색 결과가 없습니다. 검색어를 바꾸거나 score_threshold를 낮춰보세요.")
        elif total_items == 1:
            recommendations.append("유용한 결과가 1개뿐입니다. 검색어를 확장하거나 관련 키워드를 추가해보세요.")
        elif total_items < 3:
            recommendations.append("결과가 부족합니다. search_multiplier를 늘리거나 다른 검색 전략을 고려해보세요.")

        # 품질 관련 권장사항 (0.3=보통, 0.5+=높음 기준)
        if avg_final_score < 0.15:
            recommendations.append("결과 품질이 매우 낮습니다. 검색어를 더 구체적으로 작성해보세요.")
        elif avg_final_score < 0.25:
            recommendations.append("전반적인 관련도가 낮습니다. 검색어를 구체화하거나 score_threshold를 낮춰보세요.")

        # 다양성 관련 권장사항
        if diversity_score < 0.3 and total_items > 1:
            recommendations.append("결과가 특정 폴더에 집중되어 있습니다. gamma 값을 증가시켜 다양성을 높여보세요.")
        elif diversity_score < 0.5 and total_items > 2:
            recommendations.append("다양성이 낮습니다. search_multiplier를 늘려 더 넓은 범위에서 검색해보세요.")

        # 일관성 관련 권장사항
        if consistency_score < 0.5 and total_items > 1:
            recommendations.append("선택된 문서들의 품질이 일관되지 않습니다. alpha 값을 조정해보세요.")

        # 긍정적 피드백 (0.3=보통, 0.5+=높음 기준)
        if len(recommendations) == 0:
            if total_items >= 3 and avg_final_score >= 0.5:
                recommendations.append("검색 품질이 우수합니다! 현재 설정을 유지하세요.")
            elif avg_final_score >= 0.3:
                recommendations.append("검색 결과가 양호합니다.")
            else:
                recommendations.append("검색 결과가 보통 수준입니다.")

        return recommendations

    def adaptive_parameter_adjustment(self, quality_analysis: Dict[str, Any],
                                    current_params: Dict[str, float]) -> Dict[str, float]:
        """품질 분석 결과에 따른 적응적 파라미터 조정"""
        adjusted_params = current_params.copy()

        diversity_score = quality_analysis.get("diversity_score", 0.0)
        avg_final_score = quality_analysis.get("avg_final_score", 0.0)
        consistency_score = quality_analysis.get("consistency_score", 0.0)

        # 다양성이 낮으면 gamma 증가
        if diversity_score < 0.4:
            adjusted_params["gamma"] = min(0.3, current_params["gamma"] * 1.5)
            logger.info("다양성 개선을 위해 gamma를 %.2f로 조정", adjusted_params["gamma"])

        # 관련도가 낮으면 alpha 증가 (0.3=보통 기준)
        if avg_final_score < 0.2:
            adjusted_params["alpha"] = min(0.8, current_params["alpha"] * 1.2)
            logger.info("관련도 개선을 위해 alpha를 %.2f로 조정", adjusted_params["alpha"])

        # 일관성이 낮으면 beta 조정
        if consistency_score < 0.4:
            adjusted_params["beta"] = max(0.1, current_params["beta"] * 0.8)
            logger.info("일관성 개선을 위해 beta를 %.2f로 조정", adjusted_params["beta"])

        return adjusted_params
