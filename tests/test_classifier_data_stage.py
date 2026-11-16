"""Leakage/source identity, bounded diagnostic repeats and free schema inventory."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_classifier_data_stage as stage  # noqa: E402
import classifier_data_stage as runner  # noqa: E402
from configuration_splits import family_keys  # noqa: E402


def item(db="new", question="Show salaries of employees", qid=1):
    return {"db_id": db, "question_id": qid, "question": question, "SQL": "SELECT salary FROM employee"}


def old_row(split, db, qid=1, question="Show salaries of employees"):
    it = item(db, question, qid)
    return {"id": f"ambrosia:{qid}", "benchmark": "ambrosia", "db_id": db, "split": split,
            "inputs": {"question": question}, "family_keys": family_keys(it, "ambrosia")}


class Tests(unittest.TestCase):
    def test_source_ids_are_namespaced_by_content_not_bare_number(self):
        source = item("fresh", "Which artists won awards?", qid=1)
        old = [old_row("test", "protected", qid=1)]
        candidates, _ = stage.audit_pools(old, {"ambrosia": [source]})
        self.assertEqual(candidates[0]['status'], 'eligible_train_candidate')
        self.assertNotEqual(candidates[0]['id'], old[0]['id'])

    def test_holdout_database_and_family_are_both_protected(self):
        old = [old_row("dev", "protected")]
        pool = [item("protected", "Unrelated contents", 2), item("another", "Show salaries of employees", 3)]
        before = copy.deepcopy(old)
        candidates, _ = stage.audit_pools(old, {"ambrosia": pool})
        self.assertTrue(all(r['status'] == 'blocked_holdout_component' for r in candidates))
        self.assertEqual(old, before)

    def test_transitive_bridge_to_holdout_is_blocked(self):
        old = [old_row("test", "protected", question="Who lives in Boston?")]
        pool = [item("a", "Who lives in Seattle?", 2), item("a", "Who lives in Boston?", 3)]
        candidates, _ = stage.audit_pools(old, {"ambrosia": pool})
        self.assertTrue(all(r['status'] == 'blocked_holdout_component' for r in candidates))

    def test_same_input_with_conflicting_reference_is_rejected(self):
        a, b = item(), item(qid=2)
        b['SQL'] = 'SELECT other FROM employee'
        with self.assertRaises(ValueError):
            stage.audit_pools([], {'ambrosia': [a, b]})

    def test_pilot_is_bounded_deterministic_and_uses_only_eligible_new_dbs(self):
        candidates = []
        for db in ['scope_a', 'vague_b', 'attachment_c']:
            for q in range(5):
                candidates.append({'id':f'{db}:{q}', 'db_id':db, 'benchmark':'ambrosia',
                                   'new_database':True, 'status':'eligible_train_candidate'})
        selected = stage.pilot_candidates(candidates, database_limit=2)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({r['db_id'] for r in selected}), 2)
        self.assertEqual(selected, stage.pilot_candidates(list(reversed(candidates)), database_limit=2))

    def test_repeat_selection_never_uses_dev_or_test_outcomes(self):
        rows = [{'id':'bird:1','benchmark':'bird','split':'train','quality_targets':[0,1,0,1,1], 'group_id':'a'},
                {'id':'bird:2','benchmark':'bird','split':'dev','quality_targets':[0,1,0,1,1], 'group_id':'b'},
                {'id':'ambrosia:3','benchmark':'ambrosia','split':'train','quality_targets':[1]*4, 'group_id':'c'},
                {'id':'ambrosia:4','benchmark':'ambrosia','split':'test','quality_targets':[1,0,0,1], 'group_id':'d'}]
        selected = stage.select_repeats(rows, {'ambrosia:3','ambrosia:4'}, controls_per_task=0)
        self.assertEqual(set(selected['bird']), {'bird:1'})
        self.assertEqual(set(selected['ambrosia']), {'ambrosia:3'})

    def test_inventory_checks_every_reference_without_executing_sql(self):
        valid, missing, empty = item('a', qid=1), item('b', qid=2), item('c', qid=3)
        pilot = [{'db_id':v['db_id'],'item':v} for v in [valid, missing, empty]]
        snapshots = {'a': [['employee','salary','INTEGER']], 'b':None, 'c':[]}
        result = runner.inventory_snapshot(pilot, snapshots.get)
        self.assertEqual(result['approved_reference_names'], 1)
        self.assertEqual(result['availability'], {'a':'schema_present','b':'database_missing','c':'empty_schema'})
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['reference_sql_executed'])
        self.assertEqual(result['llm_calls'], 0)

    def test_unknown_or_mismatched_reference_is_not_approved(self):
        result = runner.inventory_snapshot([{'db_id':'a','item':item('a')}],
                                          lambda db: [['other','other','TEXT']])
        self.assertEqual(result['approved_reference_names'], 0)

    def test_repeat_guard_rejects_foreign_holdout_items(self):
        it = item('a')
        m = {'version':stage.VERSION,'sources_sha256':{},'jobs':[{'benchmark':'ambrosia',
              'configuration':'rules','features':[14],'repeat':2,'items':[it]}]}
        m['sha256'] = stage.data.fingerprint(m)
        with patch.object(stage.data,'load_context',return_value=(None,None,None,{'ambrosia:1':{'split':'dev','item':it}})):
            with self.assertRaises(ValueError):
                stage.validate(m)

    def test_report_averages_passes_instead_of_choosing_best(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(stage.data, 'ROOT', Path(temp)):
            root = Path(temp) / 'stage/repeats'
            folder = root / 'jobs/r2__ambrosia__rules_taxonomy'
            folder.mkdir(parents=True)
            it = item('a')
            job = {'name':'r2__ambrosia__rules_taxonomy', 'repeat':2, 'benchmark':'ambrosia',
                   'configuration':'rules_taxonomy','features':[14,32],'items':[it]}
            manifest = {'sha256':'fake', 'candidates':{'ambrosia':{'instruction':'fixed'}},
                        'selection':{'ambrosia':{'ambrosia:1':['test']}}, 'jobs':[job]}
            key = runner.collect.expected_key(manifest,job,it)
            value = {'question_id':1,'db_id':'a','prediction':'ambiguous','score':0,'error':None}
            (folder/'evaluations.json').write_text(json.dumps({key:value}),encoding='utf-8')
            source = Path(temp)/stage.data.DATA_OUTPUT
            source.mkdir(parents=True)
            observations = {n:{'score':1,'source':'initial.json','key':n} for n in stage.data.TASK_CONFIGURATIONS['ambrosia']}
            (source/'examples.jsonl').write_text(json.dumps({'id':'ambrosia:1','observations':observations})+'\n',encoding='utf-8')
            with patch.object(runner.collect,'completed',return_value=True):
                report = runner.export_repeats(root,manifest)
            anchor = next(v for v in report['records'] if v['configuration']=='rules_taxonomy')
            self.assertEqual(anchor['mean_quality_observed_passes'], .5)
            self.assertTrue(anchor['outcome_disagreement'])
            self.assertFalse(report['dataset_rewritten'])

    def test_distinct_repeat_jobs_generate_fresh_answers_and_resume_locally(self):
        import prompt_pilot_runtime as base
        import prompt_comparison_runtime as runtime
        import socket

        manifest = stage.data.read(stage.ROOT/stage.data.DEFAULT_OUTPUT/'experiment.json')
        manifest['version'] = stage.VERSION
        seen = []
        it = {'question_id':99999,'db_id':'fake_db','question':'List names','SQL':'SELECT reference_only'}

        async def prepare(session, profile):
            session.impl = SimpleNamespace(**manifest['task_manifests']['ambrosia']['fixed_prompts'])

            async def query(question):
                self.assertNotIn(it['SQL'],question)
                seen.append(question)
                return {'status':'success','query':'SELECT 1'}

            session.tools[profile] = {'fake_db':SimpleNamespace(query=query,engine=SimpleNamespace(dispose=lambda:None))}

        with tempfile.TemporaryDirectory() as temp, patch.object(base,'load_dotenv'), \
                patch.object(base,'dotenv_values',return_value={}), \
                patch.object(base.pilot,'public_model',return_value=manifest['model']), \
                patch.object(runtime.Session,'prepare_tools',prepare), \
                patch.object(base.Session,'score',return_value=(1.0,None)), \
                patch.object(base.asyncio,'sleep',new=AsyncMock()):
            root = Path(temp)
            for repeat in (2,3):
                job = {'name':f'r{repeat}__ambrosia__rules_taxonomy','repeat':repeat,'split':'train',
                       'benchmark':'ambrosia','features':[14,32],'items':[it]}
                for _ in range(2):
                    session = runner.collect.make_session(root,manifest,job)
                    try:
                        session.runner.get_loop()
                        with patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
                            session.evaluate(it,manifest['candidates']['ambrosia'],benchmark='ambrosia')
                    finally:
                        session.close()
            self.assertEqual(len(seen),2)
            self.assertEqual(stage.data.read(root/'budget.json')['examples'],2)


if __name__ == '__main__':
    unittest.main()
