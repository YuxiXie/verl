# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/reasoners/algorithm/mcts.py

import math
import itertools
import numpy as np
from tqdm import trange
from typing import Generic, Optional, NamedTuple, Union

from verl.third_party.vllm import LLM
from verl.workers.rollout.vllm_rollout.mcts.base import (
    State, Action, Example,
    SearchAlgorithm, WorldModel, SearchConfig,
)
from verl.workers.rollout.vllm_rollout.mcts.world_model import StepSubResult, StepLMAction


class MCTSConfig(NamedTuple):
    w_exp: float = 1.0
    depth_limit: int = 16
    breadth_limit: int = 8
    n_iters: int = 1024
    gamma: float = 1.0
    

class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self, 
        action: Optional[Action], 
        depth: int = 0,
        parent: "Optional[MCTSNode]" = None,
        value: float = 0.0, 
        is_terminal: bool = False,
    ):
        self.id = next(MCTSNode.id_iter)
        
        self.action = action
        self.parent = parent
        self.children: '[list[MCTSNode]]' = []
        
        self.depth = 0 if parent is None else parent.depth + 1
        assert depth == 0 or self.depth == depth, 'check depth sanity'
        
        self.value = value
        
        self.is_terminal = is_terminal
        
        self.N = 0
        self.V = 0.0
        self.Q = self.parent.V + self.r if self.parent is not None else self.r
    
    @property
    def state(self) -> list:
        state = [StepSubResult(self.action)]
        node = self.parent
        while node is not None:
            state.append(StepSubResult(node.action))
            node = node.parent
        return state[::-1]

    @property
    def r(self) -> float:
        return self.value if self.parent is None else (self.value - self.parent.value)


class MCTSResult(NamedTuple):
    tree_state: MCTSNode
    next_action_pi: list[float]
    next_action_V: list[float]
    next_action_Q: list[float]
    trace_in_each_iter: list[list[MCTSNode]] = None
    next_action_idx: int = 0
    trace_of_nodes: list[MCTSNode] = None
    cum_reward: float = None


class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, args: MCTSConfig):
        """
        MCTS algorithm
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        
        self.n_iters = args.n_iters
        self.w_exp = args.w_exp
        self.gamma = args.gamma
        self.depth_limit = args.depth_limit
        self.breadth_limit = args.breadth_limit
        
        self.policy_model = None
        self.root: Optional[MCTSNode] = None
        self.trace_in_each_iter: list[list[int]] = []
        
        self.has_correct, self.has_incorrect = False, False
        self.increase_diversity = False

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or (node.depth - self.root.depth) > self.depth_limit

    def _select(self, node: MCTSNode, node_index: int = 0) -> list[MCTSNode]:
        init_breadth_limit, breadth_limit = self.search_config.n_init_actions, self.search_config.n_actions
        if self.increase_diversity:
            init_breadth_limit, breadth_limit = self.breadth_limit, self.breadth_limit / 4
        
        path, node_indices = [], []
        while True:
            path.append(node)
            node_indices.append(node_index)
            
            if len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path, node_indices
            
            cur_breadth_limit = 1 if node.depth % self.search_config.skip_sampling != 0 else breadth_limit
            if node.depth == 0 and len(node.children) < init_breadth_limit:
                return path, node_indices   # for initial search
            if node.depth < self.search_config.depth_limit and len(node.children) < cur_breadth_limit:
                return path, node_indices   # for later search
            
            node, node_index = self._puct_select(node)

    def _puct(self, indexed_node) -> float:
        index, node = indexed_node
        return node.Q + self.w_exp * np.sqrt(node.parent.N) / (1 + node.N)
    
    def _puct_select(self, node: MCTSNode) -> MCTSNode:
        index, node = max(enumerate(node.children), key=self._puct)
        return node, index

    def _expand_and_evaluate(self, node: MCTSNode):
        if node.is_terminal:
            return
        
        if len(node.children) >= 3 and node.depth > 0:
            import ipdb; ipdb.set_trace()
        
        actions = self.search_config.get_actions(
            self.policy_model,
            node.state,
            existed_actions=[child.action for child in node.children],
        )
        if len(actions) == 0:
            return
        
        eval_scores = self.search_config.get_values(node.state, actions)
        
        for action, (value, is_terminal) in zip(actions, eval_scores):
            child = MCTSNode(
                action=action,
                depth=node.depth + 1,
                parent=node,
                value=value,
                is_terminal=is_terminal,
            )
            child.is_terminal = child.is_terminal or self.world_model.is_terminal(child.state)
            node.children.append(child)

    def _back_propagate(self, path: list[MCTSNode]):
        node = path[-1]
        node.Q = node.r + self.gamma * node.V
        node.N += 1
        for node in reversed(path[:-1]):
            node.V = sum(max(1, child.N) * child.Q for child in node.children) / sum(max(1, child.N) for child in node.children)
            node.N += 1
            node.Q = node.r + self.gamma * node.V

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        # rollout via puct selection
        path, node_indices = self._select(node, node_index=0)
        
        while not self._is_terminal_with_depth_limit(path[-1]):
            # expansion
            self._expand_and_evaluate(path[-1])
            
            if len(path[-1].children) == 0:
                break
            
            # continue rollout
            node, index = self._puct_select(path[-1])  
            continue_path, continue_node_indices = self._select(node, node_index=index)
            path.extend(continue_path)
            node_indices.extend(continue_node_indices)
        
        # backup
        self._back_propagate(path)
        
        return node_indices
    
    def search(self):
        for i in trange(self.n_iters, desc='MCTS iteration'):
            self.trace_in_each_iter.append(self.iterate(self.root))
            if (i + 1) % 16 == 0:
                exec(f'''import pickle\nwith open('mcts_rst_{i}.pkl', 'wb') as f: \n    pickle.dump(self.root, f)''')

    def check_the_labels(self, node):
        if len(node.children) == 0:
            node.correct_prob = 1 if node.is_terminal and node.value > 0 else 0
            node.incorrect_prob = 1 if node.is_terminal and node.value < 0 else 0
        else:
            children_correct_probs, children_incorrect_probs = [], []
            for child in node.children:
                _ = self.check_the_labels(child)
                children_correct_probs.append(child.correct_prob)
                children_incorrect_probs.append(child.incorrect_prob)
            node.correct_prob = sum(children_correct_probs) / len(children_correct_probs)
            node.incorrect_prob = sum(children_incorrect_probs) / len(children_incorrect_probs)
        
        return node.correct_prob * node.incorrect_prob > 0
    
    def __call__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        policy_model: LLM,
        prompt_ids: StepLMAction,
        root_node: Optional[Union[MCTSNode, int]] = None,
    ) -> MCTSResult:
        if root_node is None:
            MCTSNode.reset_id()
        
        # Initialize tree root
        self.root = root_node
        if self.root is None:
            self.root = MCTSNode(
                depth=0,
                action=prompt_ids,
            )
        
        self.policy_model = policy_model
        self.world_model = world_model
        self.search_config = search_config

        # Conduct MCTS
        self.search()
        
        # Extract tree-structured data
        
        print(self.check_the_labels(self.root))
        
        import ipdb; ipdb.set_trace()
        exec(f'''import pickle\nwith open('mcts_rst.pkl', 'wb') as f: \n    pickle.dump(self.root, f)''')
        
        ...
        
        return MCTSResult(
            ...
        )
