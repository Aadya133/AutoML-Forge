import React, { useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';
import { api, errorText, fmt } from '../api';

const qualityColor = label =>
  label === 'good' ? 'text-emerald-300' : label === 'review' ? 'text-amber-300' : 'text-rose-300';

export function QualityBadge({ score, label }) {
  if (score === null || score === undefined) return <span className="muted">n/a</span>;
  return (
    <span className={`inline-flex items-center gap-2 font-semibold ${qualityColor(label)}`}>
      {Number(score).toFixed(1)}/100 <span className="text-xs uppercase rounded bg-slate-800 px-2 py-0.5">{label}</span>
    </span>
  );
}

export default function Analytics({ models, onExplain, setNotice }) {
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [taskFilter, setTaskFilter] = useState('all');
  const [algoFilter, setAlgoFilter] = useState('all');
  const [expFilter, setExpFilter] = useState('all');

  useEffect(() => {
    Promise.all([api.get('/api/analytics'), api.get('/api/predictions?limit=20')])
      .then(([a, p]) => { setData(a.data); setHistory(p.data); })
      .catch(e => setNotice(errorText(e)));
  }, [models.length]);

  const algorithms = [...new Set(models.map(m => m.algorithm))].sort();
  const experimentIds = [...new Set(models.map(m => m.experiment_id))].sort((a, b) => b - a);
  const filtered = useMemo(() => models.filter(m =>
    (taskFilter === 'all' || m.task_type === taskFilter) &&
    (algoFilter === 'all' || m.algorithm === algoFilter) &&
    (expFilter === 'all' || String(m.experiment_id) === expFilter)
  ).sort((a, b) => (b.cv_score ?? -Infinity) - (a.cv_score ?? -Infinity)), [models, taskFilter, algoFilter, expFilter]);

  if (!data) return <p className="muted">Loading analytics…</p>;
  const statuses = Object.entries(data.experiments_by_status);
  const primary = m => m.task_type === 'classification' ? ['f1', m.metrics.f1] : ['rmse', m.metrics.rmse];

  return <div className="space-y-5">
    <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
      <div className="card"><p className="muted text-sm">Datasets</p><div className="text-3xl font-bold mt-3">{data.datasets.length}</div></div>
      <div className="card"><p className="muted text-sm">Models trained</p><div className="text-3xl font-bold mt-3">{models.length}</div></div>
      <div className="card"><p className="muted text-sm">Experiments</p><div className="mt-3 text-sm space-y-1">{statuses.length ? statuses.map(([s, n]) => <div key={s} className="flex justify-between"><span>{s}</span><b>{n}</b></div>) : <span className="muted">none yet</span>}</div></div>
      <div className="card"><p className="muted text-sm">Predictions served</p><div className="text-3xl font-bold mt-3">{data.predictions}</div></div>
    </div>

    <section className="card">
      <h2 className="font-semibold mb-3">Dataset quality</h2>
      <p className="muted text-xs mb-3">Score out of 100. Missing cells cost up to 60 points and duplicate rows up to 30. 80+ is good, 60–80 needs review.</p>
      <div className="overflow-auto"><table className="w-full text-sm"><thead><tr className="muted text-left"><th className="py-2">Dataset</th><th>Version</th><th>Rows</th><th>Columns</th><th>Quality score</th></tr></thead>
        <tbody>{data.datasets.map(d => <tr key={d.id} className="border-t border-slate-800"><td className="py-2">{d.name}</td><td>v{d.version}</td><td>{d.rows}</td><td>{d.columns}</td><td><QualityBadge score={d.quality_score} label={d.quality_label} /></td></tr>)}</tbody></table>
        {!data.datasets.length && <p className="muted">No datasets uploaded.</p>}</div>
    </section>

    {['classification', 'regression'].map(task => {
      const rows = data.algorithms.filter(a => a.task_type === task);
      if (!rows.length) return null;
      return <section className="card" key={task}>
        <h2 className="font-semibold mb-1">Algorithm performance · {task}</h2>
        <p className="muted text-xs mb-3">Average cross-validation score ({task === 'classification' ? 'weighted F1' : 'negative RMSE, closer to 0 is better'}). Purple is the voting ensemble.</p>
        <div className="grid lg:grid-cols-2 gap-5">
          <div className="h-64"><ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows}><CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="algorithm" tick={{ fill: '#94a3b8', fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={60} /><YAxis tick={{ fill: '#94a3b8' }} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
              <Bar dataKey="avg_cv_score" name="Avg CV score" isAnimationActive={false}>
                {rows.map(a => <Cell key={a.algorithm} fill={a.algorithm === 'Voting Ensemble' ? '#a78bfa' : '#22d3ee'} />)}
              </Bar></BarChart></ResponsiveContainer></div>
          <table className="w-full text-sm self-start"><thead><tr className="muted text-left"><th className="py-2">Algorithm</th><th>Runs</th><th>Best in experiment</th><th>Avg CV</th></tr></thead>
            <tbody>{rows.map(a => <tr key={a.algorithm} className="border-t border-slate-800"><td className="py-2">{a.algorithm}</td><td>{a.count}</td><td>{a.wins}</td><td>{fmt(a.avg_cv_score, 4)}</td></tr>)}</tbody></table>
        </div>
      </section>;
    })}
    {!data.algorithms.length && <section className="card"><p className="muted">Train an experiment to see algorithm statistics.</p></section>}

    <section className="card">
      <div className="flex flex-wrap gap-3 items-end justify-between mb-3">
        <h2 className="font-semibold">Model comparison</h2>
        <div className="flex flex-wrap gap-2 text-sm">
          <select className="field !w-auto" value={taskFilter} onChange={e => setTaskFilter(e.target.value)} aria-label="Task filter"><option value="all">All tasks</option><option value="classification">Classification</option><option value="regression">Regression</option></select>
          <select className="field !w-auto" value={algoFilter} onChange={e => setAlgoFilter(e.target.value)} aria-label="Algorithm filter"><option value="all">All algorithms</option>{algorithms.map(a => <option key={a}>{a}</option>)}</select>
          <select className="field !w-auto" value={expFilter} onChange={e => setExpFilter(e.target.value)} aria-label="Experiment filter"><option value="all">All experiments</option>{experimentIds.map(id => <option key={id} value={String(id)}>Experiment #{id}</option>)}</select>
        </div>
      </div>
      <div className="overflow-auto max-h-96"><table className="w-full text-sm"><thead><tr className="muted text-left"><th className="py-2">Model</th><th>Experiment</th><th>Task</th><th>CV score</th><th>Test metric</th><th></th></tr></thead>
        <tbody>{filtered.map(m => { const [k, v] = primary(m); return <tr key={m.id} className="border-t border-slate-800"><td className="py-2">{m.algorithm} <span className="muted text-xs">#{m.id} · v{m.version}</span></td><td>#{m.experiment_id}</td><td>{m.task_type}</td><td>{fmt(m.cv_score, 4)}</td><td>{k}: {fmt(v)}</td><td><button className="text-cyan-300 text-xs" onClick={() => onExplain(m.id)}>Explain →</button></td></tr>; })}</tbody></table>
        {!filtered.length && <p className="muted py-3">No models match these filters.</p>}</div>
    </section>

    <section className="card">
      <h2 className="font-semibold mb-3">Prediction history</h2>
      {history.length ? <div className="overflow-auto max-h-80"><table className="w-full text-sm"><thead><tr className="muted text-left"><th className="py-2">When</th><th>Model</th><th>User</th><th>Input</th><th>Prediction</th></tr></thead>
        <tbody>{history.map(p => <tr key={p.id} className="border-t border-slate-800 align-top"><td className="py-2 whitespace-nowrap">{p.created_at ? new Date(p.created_at).toLocaleString() : '—'}</td><td>#{p.model_id}</td><td>{p.user || 'API'}</td><td><code className="text-xs">{JSON.stringify(p.inputs[0])}{p.inputs.length > 1 ? ` +${p.inputs.length - 1}` : ''}</code></td><td>{p.outputs.map(String).join(', ')}</td></tr>)}</tbody></table></div>
        : <p className="muted">No predictions yet. Use the Models page to make one.</p>}
    </section>
  </div>;
  
}
