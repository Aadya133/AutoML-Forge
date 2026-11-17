import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, LineChart, Line, ScatterChart, Scatter, ReferenceLine, Legend } from 'recharts';
import { Download } from 'lucide-react';
import { api, errorText, fmt } from '../api';

const tooltipStyle = { background: '#0f172a', border: '1px solid #334155' };
const axis = { fill: '#94a3b8', fontSize: 11 };

function FeatureImportance({ items, method }) {
  if (!items.length) return <p className="muted">Feature importance is not available for this model.</p>;
  const data = [...items].reverse();
  return <>
    <p className="muted text-xs mb-2">Method: {method}</p>
    <div style={{ height: Math.max(220, data.length * 28) }}><ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ left: 30 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis type="number" tick={axis} /><YAxis type="category" dataKey="feature" tick={axis} width={130} />
        <Tooltip contentStyle={tooltipStyle} formatter={v => Number(v).toFixed(4)} />
        <Bar dataKey="importance" fill="#22d3ee" isAnimationActive={false} />
      </BarChart></ResponsiveContainer></div>
  </>;
}

function RocCurve({ roc, auc }) {
  const data = roc.fpr.map((x, i) => ({ fpr: x, tpr: roc.tpr[i] }));
  return <div className="h-72"><ResponsiveContainer width="100%" height="100%">
    <LineChart data={data} margin={{ bottom: 15 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
      <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={axis} label={{ value: 'False positive rate', position: 'insideBottom', offset: -8, fill: '#94a3b8', fontSize: 11 }} />
      <YAxis type="number" domain={[0, 1]} tick={axis} label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }} />
      <Tooltip contentStyle={tooltipStyle} formatter={v => Number(v).toFixed(3)} />
      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#64748b" strokeDasharray="4 4" />
      <Line isAnimationActive={false} type="stepAfter" dataKey="tpr" stroke="#22d3ee" strokeWidth={2} dot={false} name={`ROC (AUC ${fmt(auc)})`} />
      <Legend verticalAlign="top" />
    </LineChart></ResponsiveContainer></div>;
}

function ConfusionMatrix({ matrix, labels }) {
  const max = Math.max(1, ...matrix.flat());
  return <div className="overflow-auto">
    <table className="text-sm text-center">
      <thead><tr><th className="p-2 muted text-xs text-right">actual ↓ / predicted →</th>{labels.map(l => <th key={l} className="p-2 muted">{l}</th>)}</tr></thead>
      <tbody>{matrix.map((row, i) => <tr key={i}><th className="p-2 muted text-right">{labels[i]}</th>
        {row.map((v, j) => <td key={j} className="w-20 h-14 border border-slate-800 font-semibold"
          style={{ background: `rgba(${i === j ? '34,211,238' : '251,113,133'},${0.12 + 0.6 * v / max})` }}>{v}</td>)}</tr>)}</tbody>
    </table>
    <p className="muted text-xs mt-2">Diagonal cells are correct predictions.</p>
  </div>;
}

function ResidualPlot({ residuals }) {
  const data = residuals.predicted.map((p, i) => ({ predicted: p, residual: residuals.residual[i] }));
  return <div className="h-72"><ResponsiveContainer width="100%" height="100%">
    <ScatterChart margin={{ bottom: 15 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
      <XAxis type="number" dataKey="predicted" name="Predicted" tick={axis} label={{ value: 'Predicted value', position: 'insideBottom', offset: -8, fill: '#94a3b8', fontSize: 11 }} />
      <YAxis type="number" dataKey="residual" name="Residual" tick={axis} label={{ value: 'Residual (actual − predicted)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }} />
      <Tooltip contentStyle={tooltipStyle} formatter={v => Number(v).toFixed(3)} />
      <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
      <Scatter data={data} fill="#a78bfa" isAnimationActive={false} />
    </ScatterChart></ResponsiveContainer></div>;
}

export default function Explainability({ models, modelId, setModelId, setNotice }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const current = modelId || models[0]?.id;

  useEffect(() => {
    if (!current) return;
    setLoading(true);
    api.get(`/api/models/${current}/explain`).then(r => setInfo(r.data))
      .catch(e => { setInfo(null); setNotice(errorText(e)); }).finally(() => setLoading(false));
  }, [current]);

  const download = async () => {
    try {
      const r = await api.get(`/api/models/${current}/report`, { responseType: 'blob' });
      const url = URL.createObjectURL(r.data);
      const a = document.createElement('a'); a.href = url; a.download = `model_${current}_report.md`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { setNotice(errorText(e)); }
  };

  if (!models.length) return <section className="card"><p className="muted">Train a model first, then come back here to see why it predicts what it does.</p></section>;
  const plots = info?.plots || {};
  return <div className="space-y-5">
    <section className="card flex flex-col md:flex-row gap-3 md:items-end justify-between">
      <div className="flex-1"><label className="muted text-sm">Model</label>
        <select className="field mt-1" value={current} onChange={e => setModelId(Number(e.target.value))}>
          {models.map(m => <option key={m.id} value={m.id}>#{m.id} · {m.algorithm} · experiment #{m.experiment_id} ({m.task_type})</option>)}
        </select></div>
      <button className="btn2 flex items-center gap-2" onClick={download} disabled={!info}><Download size={16} /> Download evaluation report</button>
    </section>
    {loading && <p className="muted">Computing explanation…</p>}
    {info && !loading && <>
      <section className="card"><h2 className="font-semibold mb-3">Test metrics</h2>
        <div className="flex flex-wrap gap-3 text-sm">
          <span className="bg-slate-800 rounded-lg px-3 py-1.5">CV score: {fmt(info.cv_score, 4)}</span>
          {Object.entries(info.metrics).map(([k, v]) => <span key={k} className="bg-slate-800 rounded-lg px-3 py-1.5">{k}: {fmt(v)}</span>)}
        </div></section>
      <div className="grid xl:grid-cols-2 gap-5">
        <section className="card"><h2 className="font-semibold mb-3">Feature importance</h2><FeatureImportance items={info.feature_importance} method={info.method} /></section>
        {info.task_type === 'classification' ? <>
          <section className="card"><h2 className="font-semibold mb-3">Confusion matrix</h2>{plots.confusion_matrix ? <ConfusionMatrix matrix={plots.confusion_matrix} labels={plots.labels} /> : <p className="muted">Not available.</p>}</section>
          <section className="card"><h2 className="font-semibold mb-3">ROC curve</h2>{plots.roc_curve ? <RocCurve roc={plots.roc_curve} auc={info.metrics.roc_auc} /> : <p className="muted">ROC curves are shown for binary classification with probability outputs.</p>}</section>
        </> : <section className="card"><h2 className="font-semibold mb-3">Residual plot</h2>{plots.residual_plot ? <ResidualPlot residuals={plots.residual_plot} /> : <p className="muted">Not available.</p>}<p className="muted text-xs mt-2">Points spread evenly around zero mean the model has no obvious bias.</p></section>}
      </div>
    </>}
  </div>;
}
