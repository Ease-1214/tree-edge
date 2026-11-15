"""Independent blind Task/Step timing audit of A/B generation outputs."""
import json
from pathlib import Path
from agent_system.memory.experience_type_validation import validate_experience_types
from agent_system.memory.experience_updater import ExperienceUpdater
R=Path(__file__).resolve().parents[1]/'reports'
G=R/'ab_generations'
def main():
    upd=ExperienceUpdater()
    final={}
    for mode in ('baseline','revised'):
        items=[]
        for step in (5,10,15):
            data=json.loads((G/f'step{step}-{mode}-result.json').read_text())
            if not data['api_success'] or not data['parse_success']:
                raise RuntimeError('Incomplete generation '+str((step,mode)))
            for e in data['experiences']:
                items.append(dict(e, ab_step=step))
        accepted, decisions=validate_experience_types(
            items, upd._completion_with_deadline,model=upd.model,
            token_limit_param=upd.token_limit_param,
            max_completion_tokens=upd.max_completion_tokens,
        )
        details=[]
        for item, d in zip(items,decisions):
            entry=dict(step=item['ab_step'],
                       experience_type=item['experience_type'],
                       trigger=item['trigger'],
                       state_cues=item.get('state_cues'),**d)
            details.append(entry)
            print('AUDIT',mode,item['ab_step'],
                  'declared',d['declared_type'],'inferred',d['inferred_type'],
                  'accepted',d['accepted'],'reason',d['reason'],flush=True)
        final[mode]={'total':len(items),'accepted':len(accepted),'rejected':len(items)-len(accepted),'decisions':details}
    p=R/'t73_ab_generation_audit.json'
    p.write_text(json.dumps(final,ensure_ascii=False,indent=2))
    print('AUDIT_REPORT',p,flush=True)
    for key,x in final.items():
        print('AUDIT_SUMMARY',key,x['accepted'],x['total'],flush=True)
if __name__ == '__main__':main()
