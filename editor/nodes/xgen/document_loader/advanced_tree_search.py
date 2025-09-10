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

    def analyze_selection_quality(self, selected_items: List[Dict]) -> Dict[str, Any]:
        """선택 결과 품질 분석"""
        if not selected_items:
            return {}

        paths = [item.get("directory_full_path", "") for item in selected_items if item.get("directory_full_path")]
        unique_paths = set(paths)

        diversity_score = len(unique_paths) / len(selected_items) if selected_items else 0.0
        avg_score = np.mean([item.get("score", 0.0) for item in selected_items])
        avg_final_score = np.mean([item.get("final_score", 0.0) for item in selected_items])

        path_distribution = defaultdict(int)
        for path in paths:
            path_distribution[path] += 1

        return {
            "diversity_score": diversity_score,
            "unique_paths": len(unique_paths),
            "avg_original_score": avg_score,
            "avg_final_score": avg_final_score,
            "path_distribution": dict(path_distribution)
        }
