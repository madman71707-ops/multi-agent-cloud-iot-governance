import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, ArrowRight, Cpu, Gauge, Layers3, ShieldCheck, Sparkles, TrendingUp } from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API_BASE = 'http://127.0.0.1:8001'

// Fallback card metadata; replaced by GET /scenarios once the backend answers.
const SCENARIO_FALLBACK = [
  { id: 'normal_operation', name: 'Normal Operation', description: 'Baseline operation' },
  { id: 'high_network_load', name: 'High Network Load', description: 'Network stress' },
  { id: 'low_trust', name: 'Low Trust', description: 'Trust degradation' },
  { id: 'controlled_cyber_threat', name: 'Controlled Cyber Threat', description: 'Threat injection' },
  { id: 'high_resource_pressure', name: 'High Resource Pressure', description: 'Resource pressure' },
  { id: 'combined_threat_resource_pressure', name: 'Combined Threat + Resource Pressure', description: 'Combined stress conditions' },
]

// Scenario-driven simulation: each Run Simulation click advances one tick.

const navItems = ['Overview', 'Simulation', 'Agents', 'TAPNA', 'Analytics']

const TONE_TEXT = {
  red: 'border-red-500/50 bg-red-500/10 text-red-200',
  emerald: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-200',
  amber: 'border-amber-500/50 bg-amber-500/10 text-amber-200',
}
const TONE_DOT = {
  red: 'bg-red-400',
  emerald: 'bg-emerald-400',
  amber: 'bg-amber-400',
}

function statusTone(status) {
  if (status === 'THREAT_ACTIVE') return 'red'
  if (status === 'HEALTHY') return 'emerald'
  return 'amber'
}

function agentStatus(value) {
  if (value >= 0.7) return 'STRONG'
  if (value >= 0.45) return 'STABLE'
  return 'STRAINED'
}

const fmt = (value, digits = 4) => Number(value ?? 0).toFixed(digits)

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error(
      'Cannot reach the simulation service. Please try again in a moment.',
    )
  }
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = body?.detail ?? ''
    } catch {
      detail = ''
    }
    throw new Error(detail || `Simulation request failed${response.status ? ` (${response.status})` : ''}.`)
  }
  return response.json()
}

function App() {
  const [selectedScenario, setSelectedScenario] = useState('normal_operation')
  const [scenarios, setScenarios] = useState([])
  const [simulation, setSimulation] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadInitial = async () => {
    setLoading(true)
    setError('')
    try {
      const [scenarioList, state, hist] = await Promise.all([
        request('/scenarios'),
        request('/state'),
        request('/history'),
      ])
      setScenarios(scenarioList)
      setSimulation(state)
      setHistory(hist.history ?? [])
      if (state?.scenario) setSelectedScenario(state.scenario)
    } catch (err) {
      setError(err.message || 'Simulation service unavailable.')
    } finally {
      setLoading(false)
    }
  }

  const selectScenario = async (scenarioId) => {
    setSelectedScenario(scenarioId)
    setLoading(true)
    setError('')
    try {
      const state = await request(`/scenario/${scenarioId}`, { method: 'POST' })
      setSimulation(state)
      const hist = await request('/history')
      setHistory(hist.history ?? [])
    } catch (err) {
      setError(err.message || 'Scenario initialization failed.')
    } finally {
      setLoading(false)
    }
  }

  const runSimulation = async () => {
    setLoading(true)
    setError('')
    try {
      const state = await request('/run', { method: 'POST' })
      setSimulation(state)
      const hist = await request('/history')
      setHistory(hist.history ?? [])
    } catch (err) {
      setError(err.message || 'Simulation tick failed.')
    } finally {
      setLoading(false)
    }
  }

  const resetSimulation = async () => {
    setLoading(true)
    setError('')
    try {
      const state = await request('/reset', { method: 'POST' })
      setSimulation(state)
      setHistory([])
      if (state?.scenario) setSelectedScenario(state.scenario)
    } catch (err) {
      setError(err.message || 'Reset failed.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadInitial()
  }, [])

  const scenarioCards = scenarios.length ? scenarios : SCENARIO_FALLBACK
  const selectedMeta = scenarioCards.find((item) => item.id === selectedScenario)
  const lastEntry = history.length ? history[history.length - 1] : null

  const summary = useMemo(() => {
    if (!simulation) return null
    return {
      status: simulation.status ?? 'UNKNOWN',
      cpu: Number(simulation.cpu_load ?? 0),
      network: Number(simulation.network_load ?? 0),
      trust: Number(simulation.trust ?? 0),
      memory: Number(simulation.memory_load ?? 0),
      capacity: Number(simulation.capacity ?? 0),
      isolation: Boolean(simulation.isolation_status),
      attackActive: Boolean(simulation.attack_active),
      contained: Boolean(simulation.threat_contained),
      persistence: Number(simulation.attack_persistence ?? 0),
      energy: Number(simulation.energy_estimate ?? 0),
      cost: Number(simulation.cost_estimate ?? 0),
      overhead: Number(simulation.migration_overhead ?? 0),
      recommended: simulation.recommended_action ?? '—',
      wR: Number(simulation.weights?.wR ?? 0),
      wS: Number(simulation.weights?.wS ?? 0),
      wC: Number(simulation.weights?.wC ?? 0),
      tss: Number(simulation.TSS ?? 0),
      cigi: Number(simulation.CIGI ?? 0),
      uR: Number(simulation.U_R ?? 0),
      uS: Number(simulation.U_S ?? 0),
      uC: Number(simulation.U_C ?? 0),
    }
  }, [simulation])

  const chartData = useMemo(() => {
    if (!simulation) return []
    const baseline = history[0]?.before
    const points = [
      {
        tick: 0,
        TSS: Number(baseline?.TSS ?? summary?.tss ?? 0),
        wS: Number(baseline?.weights?.wS ?? summary?.wS ?? 0),
        trust_score: Number(baseline?.trust ?? summary?.trust ?? 0),
        network_load: Number(baseline?.network_load ?? summary?.network ?? 0),
        CIGI: Number(baseline?.CIGI ?? summary?.cigi ?? 0),
      },
    ]
    for (const entry of history) {
      points.push({
        tick: entry.step,
        TSS: Number(entry.TSS ?? 0),
        wS: Number(entry.weights?.wS ?? 0),
        trust_score: Number(entry.after?.trust ?? 0),
        network_load: Number(entry.after?.network_load ?? 0),
        CIGI: Number(entry.CIGI ?? 0),
      })
    }
    return points
  }, [simulation, history, summary])

  const insights = useMemo(() => {
    if (!summary) return []
    const rows = [
      summary.tss > 0.15
        ? `Threat severity ${fmt(summary.tss)} is above the governance threshold (0.15), so the security objective is prioritised.`
        : `Threat severity ${fmt(summary.tss)} is at or below the governance threshold (0.15).`,
      `Trust score is ${fmt(summary.trust, 3)} with the attack ${summary.attackActive ? 'ACTIVE' : 'inactive'} (persistence ${fmt(summary.persistence, 3)})${summary.contained ? ' and containment in progress' : ''}.`,
      `Resource pressure: CPU ${fmt(summary.cpu, 3)}, memory ${fmt(summary.memory, 3)}, network ${fmt(summary.network, 3)}, capacity ${fmt(summary.capacity, 3)}.`,
      `Estimated energy ${fmt(summary.energy, 3)} and estimated cost ${fmt(summary.cost, 3)}; migration overhead ${fmt(summary.overhead, 3)}; workload ${summary.isolation ? 'ISOLATED' : 'serving'}.`,
      `TAPNA weights wR ${fmt(summary.wR)} / wS ${fmt(summary.wS)} / wC ${fmt(summary.wC)} produced CIGI ${fmt(summary.cigi)}.`,
    ]
    if (lastEntry?.outcome) rows.push(`Last tick ${lastEntry.step} (${String(lastEntry.action).replace(/_/g, ' ')}): ${lastEntry.outcome}`)
    return rows
  }, [summary, lastEntry])

  const prevWs = lastEntry?.before?.weights?.wS ?? 1 / 3
  const wsDirection =
    summary && summary.wS > prevWs + 1e-9
      ? 'increased'
      : summary && summary.wS < prevWs - 1e-9
        ? 'decreased'
        : 'held'

  const headerStatus = loading && !simulation ? 'CONNECTING' : summary?.status ?? (error ? 'OFFLINE' : 'LOADING')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="mb-8 flex flex-col gap-6 rounded-3xl border border-slate-700 bg-slate-900/80 p-6 shadow-soft">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-400">Governance Platform</p>
              <h1 className="mt-2 text-3xl font-bold tracking-tight text-white md:text-4xl">Multi-Agent Cloud-IoT Governance</h1>
              <p className="mt-2 text-sm text-slate-300">Threat-Adaptive Digital Twin Governance Platform</p>
              <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/60 px-3 py-1 text-xs text-slate-300">
                <span className={`h-2 w-2 rounded-full ${error ? 'bg-red-400' : simulation ? 'bg-emerald-400' : 'bg-slate-500'}`} />
                {error ? 'Connection issue' : simulation ? 'Simulation live' : 'Simulation ready'}
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {navItems.map((item) => (
                <button key={item} className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-500 hover:text-white">
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.6fr_0.8fr]">
            <div className="grid-pattern rounded-2xl border border-slate-700 bg-slate-950/60 p-5">
              <div className="flex items-center gap-3 text-cyan-300">
                <Layers3 className="h-5 w-5" />
                <span className="text-sm font-semibold uppercase tracking-[0.2em]">Architecture Flow</span>
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-slate-200">
                <span className="rounded-full border border-cyan-500/60 bg-cyan-500/10 px-2.5 py-1">Digital Twin</span>
                <ArrowRight className="h-4 w-4 text-slate-400" />
                <span className="rounded-full border border-emerald-500/60 bg-emerald-500/10 px-2.5 py-1">AI Agents</span>
                <ArrowRight className="h-4 w-4 text-slate-400" />
                <span className="rounded-full border border-violet-500/60 bg-violet-500/10 px-2.5 py-1">TAPNA</span>
                <ArrowRight className="h-4 w-4 text-slate-400" />
                <span className="rounded-full border border-amber-500/60 bg-amber-500/10 px-2.5 py-1">Governance</span>
              </div>
            </div>

            <div className="card p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Current system status</p>
              <div className="mt-3 flex items-center gap-3">
                <div className={`h-3 w-3 rounded-full ${loading && !simulation ? 'animate-pulse bg-cyan-400' : TONE_DOT[statusTone(summary?.status)]}`} />
                <span className="text-lg font-semibold text-white">{headerStatus}</span>
              </div>
              <p className="mt-3 text-sm text-slate-300">
                {selectedMeta?.description ?? (loading ? 'Loading simulation scenarios...' : 'Simulation service unavailable. Please try again later.')}
              </p>
            </div>
          </div>
        </header>

        {error ? (
          <div className="mb-8 rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-red-200">
            <div className="flex items-start justify-between gap-4">
              <span>{error}</span>
              <button onClick={loadInitial} disabled={loading} className="rounded-lg border border-red-400/50 px-3 py-1 text-xs font-semibold text-red-100 hover:bg-red-500/20 disabled:opacity-60">
                Retry
              </button>
            </div>
          </div>
        ) : null}

        <main className="space-y-8">
          <section className="card p-5">
            <div className="mb-5 flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Simulation</p>
                <h2 className="mt-1 text-2xl font-semibold text-white">Scenario Control</h2>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={runSimulation}
                  className="rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={loading}
                >
                  Run Simulation
                </button>
                <button
                  onClick={resetSimulation}
                  disabled={loading}
                  className="rounded-xl border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {scenarioCards.map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => selectScenario(scenario.id)}
                  disabled={loading}
                  className={`rounded-2xl border p-4 text-left transition disabled:cursor-not-allowed disabled:opacity-70 ${selectedScenario === scenario.id ? 'border-cyan-500 bg-cyan-500/10 shadow-soft' : 'border-slate-700 bg-slate-900/70 hover:border-slate-500'}`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-white">{scenario.name}</span>
                    {selectedScenario === scenario.id && <Sparkles className="h-4 w-4 text-cyan-400" />}
                  </div>
                  <p className="mt-2 text-xs text-slate-400">{scenario.description}</p>
                </button>
              ))}
            </div>

            <p className="mt-5 text-xs text-slate-500">
              Select a scenario, then use <span className="font-semibold text-slate-300">Run Simulation</span> to advance one tick.
              {summary ? ` TAPNA currently recommends ${summary.recommended}.` : ''}
            </p>
          </section>

          {simulation && summary && (
            <>
              <section className="grid gap-5 lg:grid-cols-5">
                <MetricCard title="Memory utilization" value={summary.memory.toFixed(3)} icon={<Cpu className="h-4 w-4" />} />
                <MetricCard title="CPU utilization" value={summary.cpu.toFixed(3)} icon={<Gauge className="h-4 w-4" />} />
                <MetricCard title="Network load" value={summary.network.toFixed(3)} icon={<Activity className="h-4 w-4" />} />
                <MetricCard title="Trust score" value={summary.trust.toFixed(3)} icon={<ShieldCheck className="h-4 w-4" />} />
                <MetricCard title="System status" value={String(summary.status).replace(/_/g, ' ')} icon={<TrendingUp className="h-4 w-4" />} />
              </section>

              <section className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
                <div className="card p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-xl font-semibold text-white">AI agents</h3>
                    <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Live utility scoring</span>
                  </div>
                  <div className="grid gap-4 md:grid-cols-3">
                    <AgentCard
                      title="Resource Intelligence Agent"
                      value={summary.uR.toFixed(4)}
                      model="resource_model.joblib"
                      status={agentStatus(summary.uR)}
                      color="cyan"
                    />
                    <AgentCard
                      title="Security & Trust Agent"
                      value={summary.uS.toFixed(4)}
                      model="threat_detector.joblib"
                      status={agentStatus(summary.uS)}
                      color="red"
                      extra={`TSS ${summary.tss.toFixed(4)}`}
                    />
                    <AgentCard
                      title="Sustainability Agent"
                      value={summary.uC.toFixed(4)}
                      model="sustainability_model.joblib"
                      status={agentStatus(summary.uC)}
                      color="emerald"
                    />
                  </div>
                </div>

                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Governance</h3>
                  <div className="mt-4 space-y-3 text-sm text-slate-200">
                    <div className="flex items-center justify-between border-b border-slate-700 pb-2"><span>TAPNA status</span><span className="font-semibold text-cyan-300">Adaptive</span></div>
                    <div className="flex items-center justify-between border-b border-slate-700 pb-2"><span>wR</span><span>{summary.wR.toFixed(4)}</span></div>
                    <div className="flex items-center justify-between border-b border-slate-700 pb-2"><span>wS</span><span>{summary.wS.toFixed(4)}</span></div>
                    <div className="flex items-center justify-between border-b border-slate-700 pb-2"><span>wC</span><span>{summary.wC.toFixed(4)}</span></div>
                    <div className="flex items-center justify-between border-b border-slate-700 pb-2"><span>Selected action</span><span className="font-semibold text-white">{summary.recommended}</span></div>
                    <div className="flex items-center justify-between"><span>CIGI</span><span className="font-semibold text-amber-300">{summary.cigi.toFixed(4)}</span></div>
                  </div>
                </div>
              </section>

              <section className="grid gap-6 xl:grid-cols-2">
                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Live simulation visualization</h3>
                  <div className="mt-5 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="tssFill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="5%" stopColor="#f87171" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#f87171" stopOpacity={0.1} />
                          </linearGradient>
                          <linearGradient id="wsFill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#334155" strokeDasharray="4 4" />
                        <XAxis dataKey="tick" stroke="#94a3b8" />
                        <YAxis domain={[0, 1]} stroke="#94a3b8" />
                        <Tooltip />
                        <Legend />
                        <Area type="monotone" dataKey="TSS" stroke="#f87171" fill="url(#tssFill)" name="TSS" />
                        <Area type="monotone" dataKey="wS" stroke="#22d3ee" fill="url(#wsFill)" name="Security weight wS" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Telemtry timeline</h3>
                  <div className="mt-5 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="trustFill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="5%" stopColor="#34d399" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#34d399" stopOpacity={0.1} />
                          </linearGradient>
                          <linearGradient id="networkFill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.1} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#334155" strokeDasharray="4 4" />
                        <XAxis dataKey="tick" stroke="#94a3b8" />
                        <YAxis domain={[0, 1]} stroke="#94a3b8" />
                        <Tooltip />
                        <Legend />
                        <Area type="monotone" dataKey="trust_score" stroke="#34d399" fill="url(#trustFill)" name="Trust score" />
                        <Area type="monotone" dataKey="network_load" stroke="#fbbf24" fill="url(#networkFill)" name="Network load" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>

              <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">TAPNA decision panel</h3>
                  <div className="mt-4 space-y-4">
                    <DecisionStep label="Pareto filtering" active />
                    <DecisionStep label="TSS evaluation" active />
                    <DecisionStep label="Adaptive weights" active />
                    <DecisionStep label="Weighted selection" active />
                    <DecisionStep label="Final action" active highlight />
                  </div>
                  <div className="mt-6 rounded-2xl border border-cyan-500/40 bg-cyan-500/10 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">Selected action</p>
                        <p className="mt-1 text-2xl font-bold text-white">{summary.recommended}</p>
                      </div>
                      <div className="text-right text-sm text-slate-200">
                        <div>TSS: {summary.tss.toFixed(4)}</div>
                        <div>CIGI: {summary.cigi.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Why this decision?</h3>
                  <div className="mt-4 space-y-4 text-sm text-slate-200">
                    <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-red-200">Threat status</p>
                      <p className="mt-2">{summary.tss > 0.15 ? `Threat severity ${summary.tss.toFixed(4)} exceeded the governance threshold (0.15).` : `Threat severity ${summary.tss.toFixed(4)} stayed at or below the governance threshold (0.15).`}</p>
                    </div>
                    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-cyan-200">Weight response</p>
                      <p className="mt-2">Security weight {wsDirection} from {prevWs.toFixed(4)} to {summary.wS.toFixed(4)}.</p>
                    </div>
                    <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-emerald-200">Decision</p>
                      <p className="mt-2">TAPNA selected {summary.recommended} based on the weighted objective mix.</p>
                    </div>
                    {lastEntry?.outcome ? (
                      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3">
                        <p className="text-xs uppercase tracking-[0.2em] text-amber-200">Last outcome</p>
                        <p className="mt-2">{lastEntry.outcome}</p>
                      </div>
                    ) : null}
                  </div>
                </div>
              </section>

              <section className="card p-5">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-white">Action history</h3>
                  <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Tick {simulation.tick ?? 0} &middot; {history.length} step{history.length === 1 ? '' : 's'}</span>
                </div>
                {history.length === 0 ? (
                  <p className="text-sm text-slate-400">No simulation ticks yet. Select a scenario, then click Run Simulation.</p>
                ) : (
                  <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                    {history.map((entry) => (
                      <div key={entry.step} className="rounded-xl border border-slate-700 bg-slate-950/40 p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-sm font-semibold text-white">#{entry.step} &middot; {String(entry.action).replace(/_/g, ' ')}</span>
                          <span className="text-xs text-slate-400">
                            TSS {fmt(entry.TSS)} &middot; wS {fmt(entry.weights?.wS)} &middot; CIGI {fmt(entry.CIGI)}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-slate-300">{entry.outcome}</p>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="card p-5">
                <h3 className="text-xl font-semibold text-white">Infrastructure Insights</h3>
                <div className="mt-4 space-y-3 text-sm text-slate-200">
                  {insights.map((item, idx) => (
                    <div key={idx} className="rounded-xl border border-slate-700 bg-slate-950/40 p-3">
                      {item}
                    </div>
                  ))}
                </div>
              </section>

              <section className="grid gap-6 xl:grid-cols-[1fr_1.1fr]">
                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Agent comparison</h3>
                  <div className="mt-4 overflow-hidden rounded-xl border border-slate-700">
                    <table className="min-w-full divide-y divide-slate-700 text-left text-sm text-slate-200">
                      <thead className="bg-slate-800/80 text-slate-300">
                        <tr>
                          <th className="px-3 py-2">Agent</th>
                          <th className="px-3 py-2">Objective</th>
                          <th className="px-3 py-2">Utility</th>
                          <th className="px-3 py-2">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-700 bg-slate-900/40">
                        <tr>
                          <td className="px-3 py-2">Resource Agent</td>
                          <td className="px-3 py-2">Resource efficiency</td>
                          <td className="px-3 py-2">{summary.uR.toFixed(4)}</td>
                          <td className="px-3 py-2">{agentStatus(summary.uR)}</td>
                        </tr>
                        <tr>
                          <td className="px-3 py-2">Security Agent</td>
                          <td className="px-3 py-2">Security</td>
                          <td className="px-3 py-2">{summary.uS.toFixed(4)}</td>
                          <td className="px-3 py-2">{agentStatus(summary.uS)}</td>
                        </tr>
                        <tr>
                          <td className="px-3 py-2">Sustainability Agent</td>
                          <td className="px-3 py-2">Sustainability</td>
                          <td className="px-3 py-2">{summary.uC.toFixed(4)}</td>
                          <td className="px-3 py-2">{agentStatus(summary.uC)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="card p-5">
                  <h3 className="text-xl font-semibold text-white">Analytics</h3>
                  <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-400">CIGI by simulation tick</p>
                  <div className="mt-5 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid stroke="#334155" strokeDasharray="4 4" />
                        <XAxis dataKey="tick" stroke="#94a3b8" label={{ value: 'Tick', position: 'insideBottom', offset: -5 }} />
                        <YAxis domain={[0, 1]} stroke="#94a3b8" label={{ value: 'CIGI', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Bar dataKey="CIGI" fill="#38bdf8" radius={[4, 4, 0, 0]} name="CIGI" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

function MetricCard({ title, value, icon, status }) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between text-slate-300">
        <span className="text-xs uppercase tracking-[0.2em]">{title}</span>
        <span className="text-cyan-400">{icon}</span>
      </div>
      <div className="mt-4 flex items-center justify-between gap-3">
        <span className="text-2xl font-bold text-white">{value}</span>
        {status && (
          <span className={`badge ${TONE_TEXT[statusTone(status)]}`}>
            {status}
          </span>
        )}
      </div>
    </div>
  )
}

function AgentCard({ title, value, model, status, color, extra }) {
  const colors = {
    cyan: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200',
    red: 'border-red-500/40 bg-red-500/10 text-red-200',
    emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
  }

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-white">{title}</p>
        <span className={`rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.2em] ${colors[color]}`}>{status}</span>
      </div>
      <p className="mt-4 text-3xl font-bold text-white">{value}</p>
      <div className="mt-4 space-y-1 text-xs text-slate-300">
        <div>Model: {model}</div>
        {extra ? <div>{extra}</div> : null}
      </div>
    </div>
  )
}

function DecisionStep({ label, active, highlight }) {
  return (
    <div className={`flex items-center gap-3 rounded-xl border p-3 ${highlight ? 'border-cyan-500/60 bg-cyan-500/10' : active ? 'border-slate-600 bg-slate-900/50' : 'border-slate-700 bg-slate-900/20'} `}>
      <div className={`h-2.5 w-2.5 rounded-full ${highlight ? 'bg-cyan-400' : 'bg-slate-500'}`} />
      <span className="text-sm font-medium text-slate-100">{label}</span>
    </div>
  )
}

export default App
