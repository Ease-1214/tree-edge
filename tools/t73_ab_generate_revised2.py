"""Actual generation A/B for historical Fork windows, read-only of all existing data."""
import json
import time
import sys
from pathlib import Path
from agent_system.memory.experience_updater import ExperienceUpdater

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports'/'ab_prompts'
OUT=ROOT/'reports'/'ab_generations'
OUT.mkdir(parents=True,exist_ok=True)

def main():
    updater=ExperienceUpdater()
    # Allow selected windows via CLI, default all.
    steps = [int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [5,10,15]
    for step in steps:
        for mode in ('revised2',):
            path=P/f'step{step}-{mode}-prompt.txt'
            assert path.exists(),path
            dest=OUT/f'step{step}-{mode}-result.json'
            if dest.exists():
                print('ALREADY_COMPLETE',step,mode,flush=True)
                continue
            prompt=path.read_text()
            print('API_BEGIN',step,mode,'chars',len(prompt),flush=True)
            start=time.monotonic()
            try:
                response=updater._completion_with_deadline({
                    "model":updater.model,
                    "messages":[{"role":"user","content":prompt}],
                    updater.token_limit_param:updater.max_completion_tokens,
                })
                text=(response.choices[0].message.content or '').strip()
                parsed=updater._parse_experiences_response(text)
                success=updater.last_parse_succeeded
                d={"step":step,"mode":mode,"api_success":True,"parse_success":success,
                   "duration_seconds":round(time.monotonic()-start,2),
                   "raw":text,"experiences":parsed}
            except Exception as error:
                d={"step":step,"mode":mode,"api_success":False,"parse_success":False,
                   "duration_seconds":round(time.monotonic()-start,2),
                   "error":type(error).__name__}
            dest.write_text(json.dumps(d,ensure_ascii=False,indent=2))
            print('API_RESULT',step,mode,'success',d['api_success'],
                  'parsed',len(d.get('experiences',[])),
                  'seconds',d['duration_seconds'],flush=True)
            for item in d.get('experiences',[]):
                print('GENERATED',step,mode,item.get('experience_type'),
                      str(item.get('trigger',''))[:200],flush=True)

if __name__=='__main__':
    main()
