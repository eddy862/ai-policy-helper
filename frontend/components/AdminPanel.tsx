'use client';
import React from 'react';
import { apiIngest, apiMetrics } from '@/lib/api';

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<any>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = async () => {
    const m = await apiMetrics();
    setMetrics(m);
  };

  const ingest = async () => {
    setBusy(true);
    try {
      const res = await apiIngest();
      await refresh();
      console.log('Ingest result', res);
    } catch (e: any) {
      console.error('Error ingesting documents', e);
    }
    finally {
      setBusy(false);
    }
  };

  React.useEffect(() => { refresh(); }, []);

  return (
    <section className="admin-panel">
      <h3>Workspace</h3>
      <div className="admin-actions">
        <button className="btn btn-primary" onClick={ingest} disabled={busy}>
          {busy ? 'Indexing...' : 'Ingest sample docs'}
        </button>
        <button className="btn btn-secondary" onClick={refresh}>
          Refresh metrics
        </button>
      </div>

      {metrics && (
        <div className="metrics-box">
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
