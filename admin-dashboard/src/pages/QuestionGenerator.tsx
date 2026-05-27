import { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Slider,
  Space,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
import {
  CloudDownloadOutlined,
  FileTextOutlined,
  InfoCircleOutlined,
  InboxOutlined,
} from '@ant-design/icons';
import api from '../api/axios';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const BLOOM_OPTIONS = [
  { label: 'Remember', value: 'remember', color: 'blue', desc: 'Level 1 — Recall facts and terminology' },
  { label: 'Understand', value: 'understand', color: 'cyan', desc: 'Level 2 — Explain concepts' },
  { label: 'Apply', value: 'apply', color: 'green', desc: 'Level 3 — Solve problems' },
  { label: 'Analyze', value: 'analyze', color: 'orange', desc: 'Level 4 — Compare and break down' },
  { label: 'Evaluate', value: 'evaluate', color: 'volcano', desc: 'Level 5 — Justify and assess' },
  { label: 'Create', value: 'create', color: 'purple', desc: 'Level 6 — Design and formulate' },
];

interface FormValues {
  page_start?: number;
  page_end?: number;
  mcq_count: number;
  true_false_count: number;
  short_answer_count: number;
  fill_blank_count: number;
  essay_count: number;
  difficulty: string;
  language: string;
  title: string;
  bloom_levels: string[];
  include_answers: boolean;
}

export default function QuestionGenerator() {
  const [form] = Form.useForm<FormValues>();
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalQ, setTotalQ] = useState(13); // default counts sum

  const watchCounts = () => {
    const v = form.getFieldsValue();
    setTotalQ(
      (v.mcq_count ?? 5) +
      (v.true_false_count ?? 3) +
      (v.short_answer_count ?? 2) +
      (v.fill_blank_count ?? 3) +
      (v.essay_count ?? 0),
    );
  };

  const handleGenerate = async (values: FormValues) => {
    if (!pdfFile) {
      message.error('Please upload a PDF file first.');
      return;
    }

    const bloom = values.bloom_levels?.join(',') ?? 'remember,understand,apply';

    const formData = new FormData();
    formData.append('pdf', pdfFile);
    if (values.page_start) formData.append('page_start', String(values.page_start));
    if (values.page_end)   formData.append('page_end',   String(values.page_end));
    formData.append('mcq_count',          String(values.mcq_count ?? 5));
    formData.append('true_false_count',   String(values.true_false_count ?? 3));
    formData.append('short_answer_count', String(values.short_answer_count ?? 2));
    formData.append('fill_blank_count',   String(values.fill_blank_count ?? 3));
    formData.append('essay_count',        String(values.essay_count ?? 0));
    formData.append('difficulty',         values.difficulty ?? 'medium');
    formData.append('language',           values.language ?? 'auto');
    formData.append('title',              values.title ?? 'Exam Questions');
    formData.append('bloom_levels',       bloom);
    formData.append('include_answers',    values.include_answers ? 'true' : 'false');
    formData.append('output',             'docx');

    setLoading(true);
    try {
      const res = await api.post('/admin/generate-exam', formData, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 620_000, // 10+ min — model can be slow on first load
      });

      // Trigger browser download
      const url = URL.createObjectURL(new Blob([res.data]));
      const a   = document.createElement('a');
      const safe = (values.title ?? 'exam').replace(/[^a-zA-Z0-9_؀-ۿ ]/g, '').trim();
      a.href     = url;
      a.download = `${safe || 'exam'}.docx`;
      a.click();
      URL.revokeObjectURL(url);

      message.success('Exam generated and downloaded!');
    } catch (err: unknown) {
      // If server returned JSON error in blob form, parse it
      const blob = (err as { response?: { data?: Blob } })?.response?.data;
      if (blob instanceof Blob && blob.type === 'application/json') {
        const text = await blob.text();
        try {
          const json = JSON.parse(text);
          message.error(json.message ?? json.detail ?? 'Generation failed.');
          return;
        } catch { /* ignore */ }
      }
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 422) {
        message.error('No text found in those pages — try a different page range or check if the PDF is scanned.');
      } else if (status === 500) {
        message.error('Generation failed — Ollama may still be loading. Wait 30 s and try again.');
      } else {
        message.error('Request failed. Make sure the question generator service is running on port 8002.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Title level={4} style={{ marginBottom: 4 }}>
        Question Generator
      </Title>
      <Paragraph type="secondary" style={{ marginBottom: 24 }}>
        Upload a PDF textbook chapter → get a ready-to-print Word exam file with an answer key.
      </Paragraph>

      {/* Slow first-run notice */}
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 20 }}
        message="Generation takes 1–5 minutes"
        description="The first request after the server starts may take up to 60 s while the AI model loads. Subsequent requests are faster. Do not close the tab while generating."
      />

      <Form
        form={form}
        layout="vertical"
        onFinish={handleGenerate}
        onValuesChange={watchCounts}
        initialValues={{
          mcq_count: 5,
          true_false_count: 3,
          short_answer_count: 2,
          fill_blank_count: 3,
          essay_count: 0,
          difficulty: 'medium',
          language: 'auto',
          title: 'Exam Questions',
          bloom_levels: ['remember', 'understand', 'apply'],
          include_answers: true,
        }}
      >
        {/* ── PDF Upload ─────────────────────────────────────────────────── */}
        <Card style={{ marginBottom: 20 }}>
          <Title level={5} style={{ marginBottom: 16 }}>
            1. Upload PDF
          </Title>

          <Dragger
            accept=".pdf"
            maxCount={1}
            fileList={fileList}
            beforeUpload={(file) => {
              setPdfFile(file);
              setFileList([file]);
              return false; // prevent auto-upload
            }}
            onRemove={() => { setPdfFile(null); setFileList([]); }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ color: '#4f46e5' }} />
            </p>
            <p className="ant-upload-text">Click or drag a PDF here</p>
            <p className="ant-upload-hint">Supports Arabic and English textbooks · max 400 MB</p>
          </Dragger>

          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col xs={12}>
              <Form.Item name="page_start" label="From page (optional)">
                <InputNumber min={1} placeholder="1" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={12}>
              <Form.Item name="page_end" label="To page (optional)">
                <InputNumber min={1} placeholder="Last page" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        {/* ── Question Counts ────────────────────────────────────────────── */}
        <Card style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <Title level={5} style={{ margin: 0 }}>2. Question Types</Title>
            <Tag color="blue" style={{ fontSize: 13 }}>Total: {totalQ} questions</Tag>
          </div>

          <Row gutter={16}>
            {[
              { name: 'mcq_count',          label: 'Multiple Choice (MCQ)' },
              { name: 'true_false_count',   label: 'True / False' },
              { name: 'short_answer_count', label: 'Short Answer' },
              { name: 'fill_blank_count',   label: 'Fill in the Blank' },
              { name: 'essay_count',        label: 'Essay' },
            ].map(({ name, label }) => (
              <Col xs={24} sm={12} md={8} key={name}>
                <Form.Item name={name} label={label}>
                  <InputNumber min={0} max={50} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            ))}
          </Row>
        </Card>

        {/* ── Settings ───────────────────────────────────────────────────── */}
        <Card style={{ marginBottom: 20 }}>
          <Title level={5} style={{ marginBottom: 16 }}>3. Settings</Title>

          <Row gutter={16}>
            <Col xs={24} sm={8}>
              <Form.Item name="difficulty" label="Difficulty">
                <Select
                  options={[
                    { value: 'easy',   label: 'Easy' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'hard',   label: 'Hard' },
                  ]}
                />
              </Form.Item>
            </Col>

            <Col xs={24} sm={8}>
              <Form.Item name="language" label="Language">
                <Select
                  options={[
                    { value: 'auto', label: 'Auto Detect' },
                    { value: 'ar',   label: 'Arabic' },
                    { value: 'en',   label: 'English' },
                  ]}
                />
              </Form.Item>
            </Col>

            <Col xs={24} sm={8}>
              <Form.Item name="title" label="Exam Title">
                <Input placeholder="Exam Questions" />
              </Form.Item>
            </Col>
          </Row>

          <Divider style={{ margin: '8px 0 16px' }} />

          {/* Bloom's levels */}
          <Form.Item
            name="bloom_levels"
            label={
              <Space>
                <span>Bloom's Taxonomy Levels</span>
                <Tooltip title="Controls the cognitive depth of generated questions. Select one or more levels.">
                  <InfoCircleOutlined style={{ color: '#94a3b8' }} />
                </Tooltip>
              </Space>
            }
          >
            <Checkbox.Group style={{ width: '100%' }}>
              <Row gutter={[8, 8]}>
                {BLOOM_OPTIONS.map((opt) => (
                  <Col xs={12} sm={8} md={4} key={opt.value}>
                    <Tooltip title={opt.desc}>
                      <Checkbox value={opt.value}>
                        <Tag color={opt.color} style={{ marginLeft: 4, cursor: 'pointer' }}>
                          {opt.label}
                        </Tag>
                      </Checkbox>
                    </Tooltip>
                  </Col>
                ))}
              </Row>
            </Checkbox.Group>
          </Form.Item>

          <Form.Item name="include_answers" valuePropName="checked">
            <Checkbox>
              <Text>Include Answer Key page in the Word file</Text>
            </Checkbox>
          </Form.Item>
        </Card>

        {/* ── Submit ─────────────────────────────────────────────────────── */}
        <Button
          type="primary"
          htmlType="submit"
          icon={loading ? undefined : <CloudDownloadOutlined />}
          loading={loading}
          size="large"
          disabled={!pdfFile}
        >
          {loading ? 'Generating exam… please wait' : 'Generate & Download (.docx)'}
        </Button>

        {!pdfFile && (
          <Text type="secondary" style={{ marginLeft: 12 }}>
            Upload a PDF to enable this button
          </Text>
        )}
      </Form>
    </div>
  );
}
