import os
from os.path import join, dirname, isdir
from datasets import GraphDataset
from data_transforms import pre_transform
from train_utils import load_trainer
import argparse
import matplotlib.pyplot as plt


def check_data_graph(dataset_name):
    """ check data balance """
    dataset_kwargs = dict(input_mode='diffuse_pairwise', pre_transform=pre_transform)
    test_dataset = GraphDataset(dataset_name, **dataset_kwargs)
    for data in test_dataset:
        data = data.to('cuda')
        print(data)
        break


def evaluate_model(run_id, milestone, tries=(10, 0), json_name='eval', save_log=True,
                   run_all=False, render=True, run_only=False, resume_eval=False, render_name_extra=None,
                   return_history=False, max_timesteps=None, test_tasks=None, **kwargs):
    # only pass dataset-related overrides to load_trainer (not max_timesteps)
    loader_kwargs = {}
    if test_tasks is not None:
        loader_kwargs['test_tasks'] = test_tasks
    trainer = load_trainer(run_id, milestone, **loader_kwargs)
    if render_name_extra is not None:
        trainer.render_dir += f'_{render_name_extra}'
        if not isdir(trainer.render_dir):
            os.mkdir(trainer.render_dir)
    log = trainer.evaluate(json_name, tries=tries, render=render, save_log=save_log,
                           run_all=run_all, run_only=run_only, resume_eval=resume_eval,
                           return_history=return_history, max_timesteps=max_timesteps, **kwargs)
    return log


def compute_accuracy_from_log(log):
    """Compute an overall accuracy metric from the evaluation log."""
    if log is None:
        return 0.0
    rates = []
    for _, v in log.items():
        if isinstance(v, dict):
            if 'success_rate_top3' in v:
                rates.append(v['success_rate_top3'])
            elif 'success_rate' in v:
                rates.append(v['success_rate'])
    if len(rates) == 0:
        return 0.0
    return sum(rates) / len(rates)


def plot_accuracy_vs_timesteps(run_id, milestone, t_min=50, t_max=1000, step=50,
                               tries=(3, 0), json_name_prefix='eval_t',
                               test_tasks=None):
    """Sweep timesteps and plot accuracy vs. timesteps."""
    timesteps_list = list(range(t_min, t_max + 1, step))
    accuracies = []

    for t in timesteps_list:
        render_name_extra = f't{t}'
        log = evaluate_model(
            run_id,
            milestone,
            tries=tries,
            json_name=f'{json_name_prefix}{t}',
            save_log=False,
            render=False,
            resume_eval=False,
            render_name_extra=render_name_extra,
            visualize=False,
            max_timesteps=t,
            test_tasks=test_tasks,
        )
        acc = compute_accuracy_from_log(log)
        print(f'timesteps={t}, accuracy={acc:.3f}')
        accuracies.append(acc)

    plt.figure()
    plt.plot(timesteps_list, accuracies, marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy')
    plt.title(f'EBM Accuracy vs Timesteps (run_id={run_id}, milestone={milestone})')
    plt.grid(True)
    out_path = join(dirname(__file__), f'accuracy_vs_timesteps_{run_id}_m{milestone}.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'Saved accuracy-vs-timesteps plot to {out_path}')


MILESTONES = {
    'gtqtyd8n': 20, 'dyz4eido': 23,
    'qsd3ju74': 7, 'r26wkb13': 16,
    'ti639t00': 3, 'bo02mwbw': 11,
    '0hhtsh6a': 5, 'qcjqkout': 14,
}


def get_tests(input_mode, test_tiny=False, find_failure_cases=False):
    n = 10 if test_tiny else 100
    if input_mode == 'diffuse_pairwise':
        tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(2, 7)}
        if find_failure_cases:
            tests = {i: f"TriangularRandomSplitWorld[64]_({n})_diffuse_pairwise_test_{i}_split" for i in range(5, 7)}
    elif input_mode == 'qualitative':
        tests = {i: f'RandomSplitQualitativeWorld({n})_qualitative_test_{i}_split' for i in range(2, 6)}
    elif input_mode == 'stability_flat':
        tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object" for i in range(4, 8)}
        # tests = {i: f"RandomSplitWorld({n})_stability_flat_test_{i}_object_i=31" for i in range(7, 8)}
    elif input_mode == 'robot_box':
        tests = {i: f"TableToBoxWorld({n})_robot_box_test_{i}_object" for i in range(3, 7)}
        if find_failure_cases:
            tests = {i: f"TableToBoxWorld(10)_robot_box_test_{i}_object_i=0" for i in range(6, 7)}
    else:
        raise NotImplementedError
    if find_failure_cases:
        tests = {k: v for i, (k, v) in enumerate(tests.items()) if i >= len(tests) - 2}
    return tests


def indie_runs():
    """ for developing and debugging """
    # check_data_graph(test_tasks[4])
    # check_data_graph('TriangularRandomSplitWorld(20)_test')

    ################ sweep timesteps and plot accuracy ###########################
    eval_10_kwargs = dict(tries=(3, 0), json_name_prefix='eval_N=10_K=10_t', test_tasks={
        i: f"RandomSplitQualitativeWorld(100)_qualitative_test_{i}_split" for i in range(3, 4)
    })

    plot_accuracy_vs_timesteps('qsd3ju74', milestone=7, **eval_10_kwargs)





if __name__ == '__main__':
    indie_runs()