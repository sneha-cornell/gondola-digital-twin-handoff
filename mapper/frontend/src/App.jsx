import { Suspense, lazy, startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analyzeJob, createJob, fetchAnalyzeStatus, fetchJob, fetchJobs, fetchLayout, fetchResults, updateLayout, uploadImages } from "./api";

const SceneViewer = lazy(() => import("./components/SceneViewer"));

const TEMPLATE_LABELS = {
  gondola: "Gondola",
  wall_bay: "Wall Bay"
};
const VIEW_PRESETS = [
  { id: "iso", label: "Isometric" },
  { id: "front", label: "Front" },
  { id: "left", label: "Left" },
  { id: "top", label: "Top" }
];
const VIEW_TOGGLES = [
  { id: "wireframe", label: "Wireframe" },
  { id: "section", label: "Section" },
  { id: "xray", label: "X-Ray" },
  { id: "points", label: "Point Cloud" },
  { id: "anchors", label: "Markers" },
  { id: "rays", label: "Rays" },
  { id: "dimensions", label: "Dimensions" }
];

function formatStatusLabel(status) {
  return status
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function formatScaleSource(source) {
  return source ? formatStatusLabel(source) : "Not set";
}

function getWorkspaceGuidance(job) {
  if (!job) {
    return {
      title: "Create or choose a job",
      body: "Start with a job on the left, then upload images so the shelf layout can be reconstructed."
    };
  }

  switch (job.status) {
    case "empty":
      return {
        title: "Upload a capture set",
        body: "This job exists but has no images yet. Add photos before running analysis."
      };
    case "needs_reconstruction":
      return {
        title: "Run the analysis pass",
        body: "The images are in place. Generate results once reconstruction data is ready for this job."
      };
    case "ready_to_analyze":
      return {
        title: "Generate the shelf layout",
        body: "The text model is ready. Run analysis to fit shelves and project product anchors into the scene."
      };
    case "ready":
      return {
        title: "Review and fine-tune",
        body: "Results are loaded. Adjust width or shelf offsets if needed, then rebuild to compare changes."
      };
    default:
      return {
        title: "Keep working from the current job",
        body: "Use the controls on the left to update calibration, then inspect shelves and products in the viewer."
      };
  }
}

function getSceneCallout(results, visibleProducts, selectedShelf) {
  if (!results) {
    return {
      tone: "pending",
      kicker: "Awaiting analysis",
      title: "Build the scene before reviewing it",
      body: "Run analysis after uploading a capture set. The viewer will switch from a blank stage to a measurable rack layout.",
      meta: ["No geometry", "No shelf metrics", "No product anchors"]
    };
  }

  if (results.summary.product_count === 0) {
    return {
      tone: "caution",
      kicker: "Geometry ready",
      title: "Rack reconstruction loaded, but no products were projected",
      body: selectedShelf
        ? `The current shelf filter is active for ${selectedShelf.id}, but this run still has no mapped product anchors to inspect.`
        : "Shelf boards and rack dimensions are available, but this run did not attach any product anchors. Calibration is still available.",
      meta: [
        `${results.summary.image_count} source images`,
        `${results.summary.shelf_count} shelves detected`,
        "0 projected products"
      ]
    };
  }

  return {
    tone: "ready",
    kicker: selectedShelf ? "Shelf focus" : "Projection ready",
    title: selectedShelf
      ? `${selectedShelf.id} is filtered in the scene`
      : `${results.summary.product_count} projected products are ready to review`,
    body: selectedShelf
      ? `${visibleProducts.length} projected products remain visible in this shelf filter. Click markers in the scene or products in the list to inspect placement details.`
      : "Use the viewer presets, filters, and marker selection to inspect product placement and shelf density from the reconstructed scene.",
    meta: [
      `${results.summary.image_count} source images`,
      `${results.summary.shelf_count} shelves detected`,
      `${results.summary.product_count} projected products`
    ]
  };
}

function StatusPill({ status }) {
  return <span className={`status-pill status-${status}`}>{formatStatusLabel(status)}</span>;
}

function MiniStat({ label, value, note }) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BarRow({ label, value, total, tone = "warm", suffix = "" }) {
  const ratio = total > 0 ? Math.max(0.06, value / total) : 0;
  return (
    <div className="bar-row">
      <div className="bar-row-copy">
        <span>{label}</span>
        <strong>
          {value}
          {suffix}
        </strong>
      </div>
      <div className="bar-track">
        <div className={`bar-fill tone-${tone}`} style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatMeasure(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "n/a";
}

function formatLayoutInput(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "";
}

function createLayoutForm(layoutConfig = {}) {
  return {
    fixtureTemplate: layoutConfig.fixture_template ?? "gondola",
    referenceWidth: layoutConfig.reference_width_m == null ? "" : String(layoutConfig.reference_width_m),
    shelfHeightOffsets: Object.fromEntries(
      Object.entries(layoutConfig.shelf_height_offsets_m ?? {}).map(([shelfId, offset]) => [shelfId, String(offset)])
    )
  };
}

function buildLayoutPayload(layoutForm) {
  const shelfHeightOffsets = {};
  for (const [shelfId, value] of Object.entries(layoutForm.shelfHeightOffsets)) {
    if (value.trim() === "") {
      continue;
    }
    shelfHeightOffsets[shelfId] = Number(value);
  }

  return {
    fixture_template: layoutForm.fixtureTemplate,
    reference_width_m: layoutForm.referenceWidth.trim() === "" ? null : Number(layoutForm.referenceWidth),
    shelf_height_offsets_m: shelfHeightOffsets
  };
}

function productHasReliableName(product) {
  return Boolean(product?.product_identity_label && (product?.display_label ?? product?.label));
}

const IDENTITY_SOURCE_LABELS = {
  ocr_center_product_identifier: "Package text (OCR + catalog)",
  apple_vision_ocr: "Package text (Apple Vision OCR)",
  google_cloud_vision_ocr: "Package text (Google Vision OCR)",
  paddleocr: "Package text (PaddleOCR)",
  embedding_classifier: "Visual match (DINOv2)",
  clip_classifier: "Visual match (CLIP / SigLIP)",
  anthropic_vlm_fallback: "Vision-language model",
  anthropic_vlm_open: "Open-world VLM",
  open_world_identifier: "Open-world (Open Food Facts)",
  barcode_lookup: "Barcode → Open Food Facts",
};

function getIdentitySourceLabel(product) {
  if (!productHasReliableName(product)) {
    return "Unidentified";
  }
  const source = product?.product_identity_source;
  if (source && IDENTITY_SOURCE_LABELS[source]) {
    return IDENTITY_SOURCE_LABELS[source];
  }
  if (source) {
    return source;
  }
  return "Identified";
}

function getOpenWorldUrl(product) {
  const code = product?.off_code ?? product?.open_world?.off_code ?? product?.barcode;
  if (!code) {
    return null;
  }
  const provider = product?.open_world?.off_provider ?? "openfoodfacts";
  const subdomain = provider === "openbeautyfacts"
    ? "world.openbeautyfacts.org"
    : provider === "openproductsfacts"
      ? "world.openproductsfacts.org"
      : "world.openfoodfacts.org";
  return `https://${subdomain}/product/${encodeURIComponent(code)}`;
}

function countNamedProducts(products) {
  return products.reduce((count, product) => count + (productHasReliableName(product) ? 1 : 0), 0);
}

function getProductDisplayName(product, index) {
  if (productHasReliableName(product)) {
    return product.display_label ?? product.label ?? "Product";
  }
  return `Detected item ${String(index + 1).padStart(2, "0")}`;
}

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedJob, setSelectedJob] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedShelfId, setSelectedShelfId] = useState("");
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [viewPreset, setViewPreset] = useState("iso");
  const [viewerOptions, setViewerOptions] = useState({
    wireframe: false,
    section: false,
    xray: false,
    points: false,
    anchors: true,
    rays: true,
    dimensions: false
  });
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [availableTemplates, setAvailableTemplates] = useState(["gondola", "wall_bay"]);
  const [layoutForm, setLayoutForm] = useState(createLayoutForm());
  const [error, setError] = useState("");
  const [newJobId, setNewJobId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const shelves = results?.shelves ?? [];
  const products = results?.products ?? [];
  const layout = results?.layout ?? selectedJob?.layout_config ?? null;
  const selectedShelf = useMemo(
    () => shelves.find((shelf) => shelf.id === selectedShelfId) || null,
    [shelves, selectedShelfId]
  );
  const namedProductCount = useMemo(() => countNamedProducts(products), [products]);
  const labelsAreReliable = namedProductCount > 0;
  const productOrder = useMemo(
    () => new Map(products.map((product, index) => [product.id, index])),
    [products]
  );
  const visibleProducts = useMemo(
    () => selectedShelf
      ? products.filter((product) => product.shelf_id === selectedShelf.id)
      : products,
    [products, selectedShelf]
  );
  const visibleNamedProductCount = useMemo(() => countNamedProducts(visibleProducts), [visibleProducts]);
  const topLabels = useMemo(() => labelsAreReliable
    ? [...visibleProducts.reduce((counts, product) => {
      if (!productHasReliableName(product)) {
        return counts;
      }
      const productName = product.display_label ?? product.label ?? "Product";
      counts.set(productName, (counts.get(productName) ?? 0) + 1);
      return counts;
    }, new Map()).entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, 5)
    : [], [labelsAreReliable, visibleProducts]);
  const maxLabelCount = topLabels[0]?.[1] ?? 0;
  const averageConfidence = useMemo(() => {
    if (visibleProducts.length === 0) return 0;
    return visibleProducts.reduce((sum, product) => sum + product.confidence, 0) / visibleProducts.length;
  }, [visibleProducts]);
  const averageConfidenceLabel = visibleProducts.length > 0 ? `${Math.round(averageConfidence * 100)}%` : "n/a";
  const averageConfidenceNote = visibleProducts.length > 0 ? "visible products" : "no products visible";
  const shelfDistribution = useMemo(
    () => shelves.map((shelf) => ({ id: shelf.id, count: shelf.product_count })),
    [shelves]
  );
  const maxShelfCount = useMemo(
    () => shelfDistribution.reduce((maxCount, shelf) => Math.max(maxCount, shelf.count), 0),
    [shelfDistribution]
  );

  useEffect(() => {
    loadJobs();
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      return;
    }
    loadJob(selectedJobId);
  }, [selectedJobId]);

  async function loadJobs(preferredJobId) {
    setLoading(true);
    setError("");

    try {
      const payload = await fetchJobs();
      startTransition(() => {
        setJobs(payload.jobs);
      });

      const jobId = preferredJobId || selectedJobId || payload.jobs[0]?.job_id || "";
      setSelectedJobId(jobId);
      if (!jobId) {
        setSelectedJob(null);
        setResults(null);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadJob(jobId) {
    setError("");
    setSelectedProduct(null);
    setSelectedShelfId("");

    try {
      const [payload, layoutPayload] = await Promise.all([fetchJob(jobId), fetchLayout(jobId)]);
      setSelectedJob(payload);
      setAvailableTemplates(layoutPayload.available_templates ?? ["gondola", "wall_bay"]);
      setLayoutForm(createLayoutForm(layoutPayload.layout_config));
      if (payload.has_results) {
        const resultPayload = await fetchResults(jobId);
        setResults(resultPayload);
      } else {
        setResults(null);
      }
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function handleAnalyze() {
    if (!selectedJobId) {
      return;
    }

    setAnalyzing(true);
    setError("");
    try {
      // Kick off analysis in the background — the server returns immediately.
      await analyzeJob(selectedJobId);

      // Poll until done or error (2 s interval keeps the UI responsive).
      const jobId = selectedJobId;
      await new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const statusPayload = await fetchAnalyzeStatus(jobId);
            if (statusPayload.status === "done") {
              clearInterval(interval);
              resolve();
            } else if (statusPayload.status === "error") {
              clearInterval(interval);
              reject(new Error(statusPayload.error ?? "Analysis failed."));
            }
          } catch (pollError) {
            clearInterval(interval);
            reject(pollError);
          }
        }, 2000);
      });

      // Fetch the finished results and refresh metadata in parallel.
      const [resultPayload, jobsPayload, layoutPayload] = await Promise.all([
        fetchResults(jobId),
        fetchJobs(),
        fetchLayout(jobId),
      ]);
      setResults(resultPayload);
      setSelectedShelfId("");
      setSelectedProduct(null);
      startTransition(() => setJobs(jobsPayload.jobs));
      setAvailableTemplates(layoutPayload.available_templates ?? ["gondola", "wall_bay"]);
      setLayoutForm(createLayoutForm(layoutPayload.layout_config));
      const freshJob = jobsPayload.jobs.find((j) => j.job_id === jobId) ?? null;
      if (freshJob) setSelectedJob({ ...freshJob, results: resultPayload });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleApplyLayout() {
    await applyLayoutChanges(layoutForm);
  }

  async function applyLayoutChanges(nextLayoutForm) {
    if (!selectedJobId) {
      return;
    }

    setAnalyzing(true);
    setError("");
    try {
      setLayoutForm(nextLayoutForm);
      const savedLayout = await updateLayout(selectedJobId, buildLayoutPayload(nextLayoutForm));
      setAvailableTemplates(savedLayout.available_templates ?? ["gondola", "wall_bay"]);
      setLayoutForm(createLayoutForm(savedLayout.layout_config));

      // Kick off async analysis and poll for completion.
      const jobId = selectedJobId;
      await analyzeJob(jobId);
      await new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const statusPayload = await fetchAnalyzeStatus(jobId);
            if (statusPayload.status === "done") {
              clearInterval(interval);
              resolve();
            } else if (statusPayload.status === "error") {
              clearInterval(interval);
              reject(new Error(statusPayload.error ?? "Analysis failed."));
            }
          } catch (pollError) {
            clearInterval(interval);
            reject(pollError);
          }
        }, 2000);
      });

      const [resultPayload, jobsPayload] = await Promise.all([
        fetchResults(jobId),
        fetchJobs(),
      ]);
      setResults(resultPayload);
      setSelectedShelfId("");
      setSelectedProduct(null);
      startTransition(() => setJobs(jobsPayload.jobs));
      const freshJob = jobsPayload.jobs.find((j) => j.job_id === jobId) ?? null;
      if (freshJob) setSelectedJob({ ...freshJob, results: resultPayload });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleViewportReferenceWidthCommit(referenceWidth) {
    if (!Number.isFinite(referenceWidth)) {
      return;
    }

    const nextLayoutForm = {
      ...layoutForm,
      referenceWidth: formatLayoutInput(referenceWidth)
    };
    await applyLayoutChanges(nextLayoutForm);
  }

  async function handleViewportShelfOffsetCommit(shelfId, offset) {
    if (!Number.isFinite(offset)) {
      return;
    }

    const nextLayoutForm = {
      ...layoutForm,
      shelfHeightOffsets: {
        ...layoutForm.shelfHeightOffsets,
        [shelfId]: formatLayoutInput(offset)
      }
    };
    await applyLayoutChanges(nextLayoutForm);
  }

  const canAnalyze = Boolean(selectedJobId) && selectedJob?.status !== "empty";
  const activeTemplateLabel = TEMPLATE_LABELS[layout?.fixture_template] ?? "Unset";
  const scaleSourceLabel = formatScaleSource(layout?.scale_source);
  const activeScale = Number.isFinite(layout?.scale_factor) ? `${layout.scale_factor.toFixed(3)}x` : "n/a";
  const workspaceGuidance = getWorkspaceGuidance(selectedJob);
  const selectedProductName = selectedProduct
    ? getProductDisplayName(selectedProduct, productOrder.get(selectedProduct.id) ?? 0)
    : "";
  const namingGuidance = namedProductCount > 0
    ? `Exact names were read from package text for ${namedProductCount} of ${products.length} projected products. Unreadable products keep neutral labels.`
    : results?.labeling?.product_name_source
      ? "Package text recognition is enabled for this run, but exact names were not readable with enough confidence."
      : "This run uses generic object detection only, so unreadable products stay neutral instead of showing false names.";
  const detailTitle = selectedProduct ? "Product details" : selectedShelf ? "Shelf details" : "Selection details";
  const detailCopy = selectedProduct
    ? `${selectedProductName} projected onto ${selectedProduct.shelf_id}.`
    : selectedShelf
      ? `${selectedShelf.id} currently contains ${selectedShelf.product_count} projected products.`
      : "Click a shelf board or product placement in the viewer to inspect it here.";
  const visibleProductsCopy = selectedShelf
    ? `Showing projected products on ${selectedShelf.id}. ${namingGuidance}`
    : `Showing every projected product in the current scene. ${namingGuidance}`;
  const analysisCopy = namedProductCount > 0
    ? "Recognized product names and shelf density for the current filter. Unnamed products are excluded from name counts."
    : results?.labeling?.product_name_source
      ? "Shelf density is available, but package text was not readable enough to promote exact names in the current filter."
      : "Shelf density is available, but the current run does not have a product naming stage.";
  const viewerCaption = results
    ? selectedShelf
      ? `${selectedShelf.id} board with ${visibleProducts.length} projected products`
      : `${results.summary.shelf_count} shelves with ${results.summary.product_count} projected products`
    : "Run analysis to build the shelf layout and projected products.";
  const sceneCallout = getSceneCallout(results, visibleProducts, selectedShelf);

  const handleSelectShelf = useCallback((shelfId) => {
    setSelectedShelfId((currentShelfId) => (currentShelfId === shelfId ? "" : shelfId));
    setSelectedProduct((currentProduct) => {
      if (!currentProduct) {
        return null;
      }
      return currentProduct.shelf_id === shelfId ? currentProduct : null;
    });
  }, []);

  const handleSelectProduct = useCallback((product) => {
    setSelectedProduct(product);
    setSelectedShelfId(product.shelf_id);
  }, []);

  async function handleCreateJob() {
    const trimmedId = newJobId.trim();
    if (!trimmedId) return;
    setLoading(true);
    setError("");
    try {
      await createJob(trimmedId);
      setNewJobId("");
      await loadJobs(trimmedId);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleUploadImages(files) {
    if (!selectedJobId || files.length === 0) return;
    setUploading(true);
    setUploadError("");
    try {
      await uploadImages(selectedJobId, files);
      await loadJob(selectedJobId);
      await loadJobs(selectedJobId);
    } catch (requestError) {
      setUploadError(requestError.message);
    } finally {
      setUploading(false);
    }
  }

  function handleDragOver(event) {
    event.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave() {
    setDragOver(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragOver(false);
    const files = [...event.dataTransfer.files].filter((f) => /\.(jpe?g|png)$/i.test(f.name));
    if (files.length) handleUploadImages(files);
  }

  function handleFileInputChange(event) {
    const files = [...event.target.files];
    if (files.length) handleUploadImages(files);
    event.target.value = "";
  }

  function handleLayoutTemplateChange(value) {
    setLayoutForm((current) => ({
      ...current,
      fixtureTemplate: value
    }));
  }

  function handleReferenceWidthChange(value) {
    setLayoutForm((current) => ({
      ...current,
      referenceWidth: value
    }));
  }

  function handleShelfOffsetChange(shelfId, value) {
    setLayoutForm((current) => ({
      ...current,
      shelfHeightOffsets: {
        ...current.shelfHeightOffsets,
        [shelfId]: value
      }
    }));
  }

  function toggleViewerOption(optionId) {
    setViewerOptions((current) => ({
      ...current,
      [optionId]: !current[optionId]
    }));
  }

  return (
    <div className="app-shell">
      <div className="app-frame">
        <header className="topbar">
          <div className="brand-block">
            <p className="eyebrow">Shelf review workspace</p>
            <h1>Shelf Mapper</h1>
            <p className="brand-copy">
              Upload a capture set, calibrate the fixture, and review the reconstructed shelf layout in one place.
            </p>
          </div>

          <div className="workspace-note">
            <span className="note-label">Next step</span>
            <strong>{workspaceGuidance.title}</strong>
            <p>{workspaceGuidance.body}</p>
          </div>

          <div className="topbar-actions">
            <button className="ghost-button" type="button" onClick={() => loadJobs(selectedJobId)}>
              Refresh data
            </button>
            <button className="primary-button" type="button" onClick={handleAnalyze} disabled={!canAnalyze || analyzing}>
              {analyzing ? "Analyzing..." : "Analyze job"}
            </button>
          </div>
        </header>

        <section className="summary-strip">
          <MiniStat
            label="Active job"
            value={selectedJobId || "None"}
            note={selectedJob ? formatStatusLabel(selectedJob.status) : "no job selected"}
          />
          <MiniStat label="Template" value={activeTemplateLabel} note={scaleSourceLabel} />
          <MiniStat label="Shelves" value={results?.summary.shelf_count ?? "—"} note="detected boards" />
          <MiniStat
            label="Visible products"
            value={results ? visibleProducts.length : "—"}
            note={selectedShelf ? selectedShelf.id : "entire scene"}
          />
        </section>

        {error ? <div className="banner banner-error">{error}</div> : null}
        {loading ? <div className="banner banner-muted">Loading jobs and layout data...</div> : null}

        <main className="workspace-grid">
          <aside className="rail rail-left">
            <section className="panel-card">
              <div className="panel-header">
                <div>
                  <p className="section-kicker">Step 1</p>
                  <h2>Choose a job</h2>
                  <p className="panel-copy">Create a workspace, pick the active job, and upload the source images.</p>
                </div>
              </div>

              <div className="new-job-form">
                <label className="field">
                  <span>New Job ID</span>
                  <input
                    id="new-job-id"
                    name="new-job-id"
                    type="text"
                    value={newJobId}
                    onChange={(event) => setNewJobId(event.target.value)}
                    onKeyDown={(event) => { if (event.key === "Enter") handleCreateJob(); }}
                    placeholder="e.g. aisle-3-gondola"
                    disabled={loading}
                  />
                </label>
                <button
                  className="ghost-button apply-button"
                  type="button"
                  onClick={handleCreateJob}
                  disabled={!newJobId.trim() || loading}
                >
                  Create job
                </button>
              </div>

              <label className="field">
                <span>Current job</span>
                <select
                  id="active-job"
                  name="active-job"
                  value={selectedJobId}
                  onChange={(event) => setSelectedJobId(event.target.value)}
                  disabled={jobs.length === 0}
                >
                  {jobs.length === 0 && <option value="">No jobs discovered</option>}
                  {jobs.map((job) => (
                    <option key={job.job_id} value={job.job_id}>
                      {job.job_id}
                    </option>
                  ))}
                </select>
              </label>

              <div className="chip-row">
                {selectedJob ? <StatusPill status={selectedJob.status} /> : null}
                <span className="soft-chip">{selectedJob?.image_count ?? 0} images</span>
                <span className="soft-chip">{selectedJob?.has_text_model ? "Text model ready" : "Text model missing"}</span>
                <span className="soft-chip">{selectedJob?.colmap_available ? "COLMAP available" : "COLMAP not local"}</span>
              </div>

              {selectedJobId ? (
                <>
                  <input
                    ref={fileInputRef}
                    id="job-image-upload"
                    name="job-image-upload"
                    type="file"
                    accept=".jpg,.jpeg,.png"
                    multiple
                    style={{ display: "none" }}
                    onChange={handleFileInputChange}
                  />
                  <div
                    className={`upload-zone${dragOver ? " drag-over" : ""}${uploading ? " uploading" : ""}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !uploading && fileInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInputRef.current?.click(); }}
                  >
                    <span className="upload-icon">Upload</span>
                    <span>{uploading ? "Uploading images..." : "Drop images here or click to upload"}</span>
                    <small>.jpg · .jpeg · .png</small>
                  </div>
                  {uploadError ? <div className="banner banner-error">{uploadError}</div> : null}
                </>
              ) : null}
            </section>

            <section className="panel-card">
              <div className="panel-header">
                <div>
                  <p className="section-kicker">Step 2</p>
                  <h2>Calibrate the layout</h2>
                  <p className="panel-copy">Use one known width to scale the rack, then fine-tune shelf heights if needed.</p>
                </div>
              </div>

              <div className="layout-form">
                <label className="field">
                  <span>Fixture Template</span>
                  <select
                    id="fixture-template"
                    name="fixture-template"
                    value={layoutForm.fixtureTemplate}
                    onChange={(event) => handleLayoutTemplateChange(event.target.value)}
                    disabled={!selectedJobId}
                  >
                    {availableTemplates.map((template) => (
                      <option key={template} value={template}>
                        {TEMPLATE_LABELS[template] ?? template}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="field">
                  <span>Reference Width (m)</span>
                  <input
                    id="reference-width"
                    name="reference-width"
                    type="number"
                    step="0.01"
                    min="0"
                    value={layoutForm.referenceWidth}
                    onChange={(event) => handleReferenceWidthChange(event.target.value)}
                    placeholder="1.25"
                    disabled={!selectedJobId}
                  />
                </label>

                <p className="support-copy">
                  Tip: once results are loaded, you can also drag the width and shelf handles directly in the viewer.
                </p>

                {shelves.length > 0 ? (
                  <div className="offset-section">
                    <div className="panel-subheader">
                      <h3>Shelf Offsets</h3>
                      <span>meters</span>
                    </div>

                    <div className="offset-grid">
                      {shelves.map((shelf) => (
                        <label key={shelf.id} className="field compact-field">
                          <span>{shelf.id}</span>
                          <input
                            id={`shelf-offset-${shelf.id}`}
                            name={`shelf-offset-${shelf.id}`}
                            type="number"
                            step="0.01"
                            value={layoutForm.shelfHeightOffsets[shelf.id] ?? ""}
                            onChange={(event) => handleShelfOffsetChange(shelf.id, event.target.value)}
                            placeholder="0.00"
                            disabled={!selectedJobId}
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                ) : null}

                <button
                  className="ghost-button apply-button"
                  type="button"
                  onClick={handleApplyLayout}
                  disabled={!canAnalyze || analyzing}
                >
                  {analyzing ? "Saving..." : "Save calibration and rebuild"}
                </button>
              </div>
            </section>

            <section className="panel-card">
              <div className="panel-header">
                <div>
                  <p className="section-kicker">Step 3</p>
                  <h2>Browse shelves</h2>
                  <p className="panel-copy">Use the list to focus the viewer on a single shelf and its projected products.</p>
                </div>
                {selectedShelf ? (
                  <button className="ghost-button mini-button" type="button" onClick={() => setSelectedShelfId("")}>
                    Clear
                  </button>
                ) : null}
              </div>

              {shelves.length > 0 ? (
                <div className="shelf-list">
                  {shelves.map((shelf) => (
                    <button
                      key={shelf.id}
                      type="button"
                      className={`shelf-chip ${selectedShelf?.id === shelf.id ? "active" : ""}`}
                      onClick={() => handleSelectShelf(shelf.id)}
                    >
                      <span>{shelf.id}</span>
                      <strong>{shelf.product_count}</strong>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="empty-copy">Run analysis to list the detected shelf boards.</p>
              )}
            </section>
          </aside>

          <section className="workspace-main">
            <section className="viewer-card viewer-hero">
              <div className="viewer-header">
                <div className="viewer-title-block">
                  <p className="section-kicker">Step 4</p>
                  <h2>Review the 3D scene</h2>
                  <p className="viewer-caption">{viewerCaption}</p>
                  <p className="viewer-hint">Drag to orbit, scroll to zoom, and click a shelf board or marker to inspect it.</p>
                  <div className="scene-legend">
                    <span><i className="legend-swatch legend-rack" /> Rack</span>
                    <span><i className="legend-swatch legend-shelf" /> Shelf</span>
                    <span><i className="legend-swatch legend-points" /> Point cloud</span>
                    <span><i className="legend-swatch legend-anchor" /> Marker</span>
                    <span><i className="legend-swatch legend-ray" /> Ray</span>
                  </div>
                </div>

                <div className="viewer-toolbar">
                  <div className="viewer-badges">
                    <span>Template {activeTemplateLabel}</span>
                    <span>Width {formatMeasure(layout?.measured_width)}</span>
                    <span>Depth {formatMeasure(layout?.measured_depth)}</span>
                    <span>Scale {activeScale}</span>
                  </div>
                  <div className="viewer-toolsets">
                    <div className="view-mode-group">
                      {VIEW_PRESETS.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          className={`tool-button ${viewPreset === preset.id ? "active" : ""}`}
                          onClick={() => setViewPreset(preset.id)}
                        >
                          {preset.label}
                        </button>
                      ))}
                    </div>
                    <div className="view-mode-group">
                      {VIEW_TOGGLES.map((toggle) => (
                        <button
                          key={toggle.id}
                          type="button"
                          className={`tool-button ${viewerOptions[toggle.id] ? "active" : ""}`}
                          onClick={() => toggleViewerOption(toggle.id)}
                        >
                          {toggle.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="viewer-stage-shell">
                <div className={`scene-callout tone-${sceneCallout.tone}`}>
                  <div className="scene-callout-copy">
                    <span className="scene-callout-kicker">{sceneCallout.kicker}</span>
                    <strong>{sceneCallout.title}</strong>
                    <p>{sceneCallout.body}</p>
                  </div>
                  <div className="scene-callout-meta">
                    {sceneCallout.meta.map((entry) => (
                      <span key={entry}>{entry}</span>
                    ))}
                  </div>
                </div>
                <div className="viewer-stage">
                  <Suspense fallback={<div className="viewer-loading">Loading 3D viewer...</div>}>
                    <SceneViewer
                      results={results}
                      layoutConfig={results?.layout ?? null}
                      selectedProduct={selectedProduct}
                      selectedShelfId={selectedShelf?.id ?? ""}
                      viewPreset={viewPreset}
                      viewerOptions={viewerOptions}
                      viewportEditingDisabled={analyzing || !selectedJobId}
                      onSelectProduct={handleSelectProduct}
                      onSelectShelf={handleSelectShelf}
                      onCommitReferenceWidth={handleViewportReferenceWidthCommit}
                      onCommitShelfOffset={handleViewportShelfOffsetCommit}
                    />
                  </Suspense>
                </div>
              </div>

              {results ? (
                <div className="viewer-footer">
                  <MiniStat label="Images" value={results.summary.image_count} note="source photos" />
                  <MiniStat label="Points" value={results.summary.point_count} note="reconstruction points" />
                  <MiniStat label="Scale source" value={scaleSourceLabel} note={activeTemplateLabel} />
                  <MiniStat label="Results" value={selectedJob?.has_results ? "Ready" : "Missing"} note="saved for this job" />
                </div>
              ) : null}
            </section>

            <section className="inspector-grid">
              <section className="panel-card panel-stretch">
                <div className="panel-header">
                  <div>
                    <p className="section-kicker">Details</p>
                    <h2>{detailTitle}</h2>
                    <p className="panel-copy">{detailCopy}</p>
                  </div>
                </div>

                {selectedProduct ? (
                <div className="focus-stack">
                  {selectedProduct.crop_url ? (
                    <img src={selectedProduct.crop_url} alt={selectedProductName} className="product-image" />
                  ) : (
                    <div className="product-image fallback-image">No crop available</div>
                  )}
                  {selectedProduct.reference_image_url ? (
                    <a
                      href={getOpenWorldUrl(selectedProduct) ?? selectedProduct.reference_image_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="reference-image-link"
                      title="Open Food Facts reference image"
                    >
                      <img
                        src={selectedProduct.reference_image_url}
                        alt={`${selectedProductName} reference`}
                        className="reference-image"
                        loading="lazy"
                      />
                      <span className="reference-image-caption">
                        Open Food Facts reference
                      </span>
                    </a>
                  ) : null}
                  <div className="info-grid">
                    <InfoRow label="Displayed name" value={selectedProductName} />
                    {selectedProduct.brand ? (
                      <InfoRow label="Brand" value={selectedProduct.brand} />
                    ) : null}
                    <InfoRow label="Shelf" value={selectedProduct.shelf_id} />
                    <InfoRow label="Confidence" value={formatPercent(selectedProduct.confidence)} />
                    <InfoRow
                      label="Identified by"
                      value={getIdentitySourceLabel(selectedProduct)}
                    />
                    {selectedProduct.off_code ? (
                      <InfoRow
                        label="Open Food Facts"
                        value={
                          <a
                            href={getOpenWorldUrl(selectedProduct) ?? "#"}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {selectedProduct.off_code}
                          </a>
                        }
                      />
                    ) : null}
                    {selectedProduct.barcode && selectedProduct.barcode !== selectedProduct.off_code ? (
                      <InfoRow label="Barcode" value={selectedProduct.barcode} />
                    ) : null}
                    {selectedProduct.open_world?.match_score != null ? (
                      <InfoRow
                        label="OFF match score"
                        value={formatPercent(selectedProduct.open_world.match_score)}
                      />
                    ) : null}
                    <InfoRow label="Source image" value={selectedProduct.image_name} />
                    <InfoRow label="World position" value={selectedProduct.p3d.map((value) => value.toFixed(2)).join(", ")} />
                  </div>
                  </div>
                ) : selectedShelf ? (
                  <div className="focus-stack">
                    <div className="info-grid">
                      <InfoRow label="Width" value={formatMeasure(selectedShelf.extents.width)} />
                      <InfoRow label="Depth" value={formatMeasure(selectedShelf.extents.depth)} />
                      <InfoRow label="Thickness" value={formatMeasure(selectedShelf.extents.thickness)} />
                      <InfoRow label="Anchor Height" value={formatMeasure(selectedShelf.geometry.anchor_height)} />
                    </div>

                  <div className="tag-cloud">
                    {selectedShelf.inventory.labels.length > 0 ? (
                      selectedShelf.inventory.labels.map((entry) => (
                        <span key={entry.label} className="info-tag">
                          {entry.label} {entry.count}
                        </span>
                      ))
                    ) : results?.labeling?.product_name_source ? (
                      <p className="empty-copy">No exact package names were readable for the projected products in this shelf.</p>
                    ) : (
                      <p className="empty-copy">Product names are unavailable because this run only has generic object detections.</p>
                    )}
                  </div>
                </div>
                ) : (
                  <div className="focus-stack">
                    <div className="info-grid">
                      <InfoRow label="Job" value={selectedJobId || "—"} />
                      <InfoRow label="Measured width" value={formatMeasure(layout?.measured_width)} />
                      <InfoRow label="Measured depth" value={formatMeasure(layout?.measured_depth)} />
                      <InfoRow label="Scale source" value={scaleSourceLabel} />
                    </div>
                  </div>
                )}
              </section>

              <section className="panel-card panel-stretch">
                <div className="panel-header">
                  <div>
                    <p className="section-kicker">Analysis</p>
                    <h2>Current mix</h2>
                    <p className="panel-copy">{analysisCopy}</p>
                  </div>
                </div>

                <div className="analytics-stack">
                  <div className="analytics-summary">
                    <MiniStat
                      label={namedProductCount > 0 ? "Named" : "Names"}
                      value={namedProductCount > 0 ? visibleNamedProductCount : results?.labeling?.product_name_source ? "0" : "Hidden"}
                      note={namedProductCount > 0 ? "visible products" : results?.labeling?.product_name_source ? "OCR enabled" : "generic detector"}
                    />
                    <MiniStat label="Avg. confidence" value={averageConfidenceLabel} note={averageConfidenceNote} />
                  </div>

                  <div className="analytics-block">
                    <div className="panel-subheader">
                      <h3>{namedProductCount > 0 ? "Recognized products" : "Product naming"}</h3>
                      <span>{namedProductCount > 0 ? "current filter" : "package text status"}</span>
                    </div>
                    {topLabels.length > 0 ? (
                      <div className="bar-list">
                        {topLabels.map(([label, count], index) => (
                          <BarRow
                            key={label}
                            label={label}
                            value={count}
                            total={maxLabelCount}
                            tone={index % 2 === 0 ? "warm" : "cool"}
                          />
                        ))}
                      </div>
                    ) : results?.labeling?.product_name_source ? (
                      <p className="empty-copy">No exact product names were readable for the current selection.</p>
                    ) : (
                      <p className="empty-copy">This run does not have package-text naming enabled, so only neutral product labels are shown.</p>
                    )}
                  </div>

                  <div className="analytics-block">
                    <div className="panel-subheader">
                      <h3>Shelf density</h3>
                      <span>products per board</span>
                    </div>
                    {shelfDistribution.length > 0 ? (
                      <div className="bar-list">
                        {shelfDistribution.map((shelf, index) => (
                          <BarRow
                            key={shelf.id}
                            label={shelf.id}
                            value={shelf.count}
                            total={maxShelfCount}
                            tone={index % 2 === 0 ? "cool" : "warm"}
                          />
                        ))}
                      </div>
                    ) : (
                      <p className="empty-copy">No shelf metrics available yet.</p>
                    )}
                  </div>
                </div>
              </section>

              <section className="panel-card panel-stretch">
                <div className="panel-header">
                  <div>
                    <p className="section-kicker">Products</p>
                    <h2>Visible products</h2>
                    <p className="panel-copy">{visibleProductsCopy}</p>
                  </div>
                  <span className="count-chip">{visibleProducts.length}</span>
                </div>

                {visibleProducts.length > 0 ? (
                  <div className="product-grid">
                    {visibleProducts.map((product) => (
                      <button
                        key={product.id}
                        type="button"
                        className={`product-chip ${selectedProduct?.id === product.id ? "active" : ""}`}
                        onClick={() => handleSelectProduct(product)}
                      >
                        <span>{getProductDisplayName(product, productOrder.get(product.id) ?? 0)}</span>
                        <small>
                          {product.shelf_id} · {formatPercent(product.confidence)}
                        </small>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="empty-copy">No product placements are available for the current filter.</p>
                )}
              </section>
            </section>
          </section>
        </main>
      </div>
    </div>
  );
}
