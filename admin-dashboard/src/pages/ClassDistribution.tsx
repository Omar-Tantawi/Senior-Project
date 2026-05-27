import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Form,
  InputNumber,
  Row,
  Select,
  Slider,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  CheckCircleOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import api from '../api/axios';

const { Title, Text } = Typography;

interface SchoolYear {
  schoolyearid: number;
  name: string;
}

interface SchoolClass {
  class_id: number;
  name: string;
  schoolyearid: number;
}

interface CriteriaWeights {
  academic: number;
  behavior: number;
  attendance: number;
}

interface BalanceMetrics {
  balance_score: number;
  is_balanced: boolean;
  explanation: string;
}

interface StudentInfo {
  name: string;
  email: string;
}

interface DistributionResult {
  distribution_id: string;
  classes: Record<string, string[]>;
  balance_metrics: BalanceMetrics;
  total_students: number;
  num_classes: number;
  students_map: Record<string, StudentInfo>;
}

interface FormValues {
  school_year_id: number;
  class_id: number;
  num_classes: number;
}

function BalanceBadge({ score, is_balanced }: { score: number; is_balanced: boolean }) {
  const color = score >= 0.85 ? 'success' : score >= 0.7 ? 'warning' : 'error';
  const label = is_balanced ? 'Fair Distribution' : 'Unbalanced';
  const icon = is_balanced ? <CheckCircleOutlined /> : <WarningOutlined />;

  return (
    <Space align="center" size={12}>
      <Badge
        color={color === 'success' ? '#16a34a' : color === 'warning' ? '#f59e0b' : '#dc2626'}
        text={
          <Text strong style={{ fontSize: 28 }}>
            {(score * 100).toFixed(1)}%
          </Text>
        }
      />
      <Tag
        color={color}
        icon={icon}
        style={{ fontSize: 14, padding: '4px 12px', borderRadius: 999 }}
      >
        {label}
      </Tag>
    </Space>
  );
}

function exportCsv(
  classes: Record<string, string[]>,
  studentsMap: Record<string, StudentInfo>,
  className: string,
) {
  const target = className === 'all' ? null : className;
  const rows: string[][] = [['Class', 'Student ID', 'Name', 'Email']];

  const entries = target
    ? [[target, classes[target]] as [string, string[]]]
    : (Object.entries(classes) as [string, string[]][]);

  for (const [cls, ids] of entries) {
    for (const id of ids) {
      const info = studentsMap[id] ?? { name: '', email: '' };
      rows.push([cls, id, info.name, info.email]);
    }
  }

  const csv = rows.map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `distribution-${target ?? 'all'}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ClassDistribution() {
  const [form] = Form.useForm<FormValues>();
  const [schoolYears, setSchoolYears] = useState<SchoolYear[]>([]);
  const [classes, setClasses] = useState<SchoolClass[]>([]);
  const [loadingYears, setLoadingYears] = useState(false);
  const [loadingClasses, setLoadingClasses] = useState(false);
  const [distributing, setDistributing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [result, setResult] = useState<DistributionResult | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  // Weight state (percentages 0-100, user adjusts freely)
  const [weights, setWeights] = useState<{ academic: number; behavior: number; attendance: number }>({
    academic: 60,
    behavior: 30,
    attendance: 10,
  });
  const weightSum = weights.academic + weights.behavior + weights.attendance;

  const fetchSchoolYears = useCallback(async () => {
    setLoadingYears(true);
    try {
      const res = await api.get('/admin/school-years');
      setSchoolYears(res.data);
    } catch {
      message.error('Failed to load school years');
    } finally {
      setLoadingYears(false);
    }
  }, []);

  const fetchClasses = useCallback(async (schoolYearId: number) => {
    setLoadingClasses(true);
    setClasses([]);
    form.setFieldValue('class_id', undefined);
    try {
      const res = await api.get('/admin/classes', { params: { schoolyearid: schoolYearId } });
      setClasses(res.data);
    } catch {
      message.error('Failed to load classes');
    } finally {
      setLoadingClasses(false);
    }
  }, [form]);

  useEffect(() => {
    fetchSchoolYears();
  }, [fetchSchoolYears]);

  const normalizedWeights = (): CriteriaWeights => {
    const sum = weightSum || 100;
    return {
      academic: weights.academic / sum,
      behavior: weights.behavior / sum,
      attendance: weights.attendance / sum,
    };
  };

  const handleDistribute = async (values: FormValues) => {
    if (weightSum === 0) {
      message.error('Weights must not all be zero.');
      return;
    }

    setDistributing(true);
    setResult(null);
    setConfirmed(false);

    try {
      const res = await api.post('/admin/distribute-classes', {
        class_id: values.class_id,
        num_classes: values.num_classes,
        criteria_weights: normalizedWeights(),
      });
      setResult(res.data);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        'Distribution failed.';
      message.error(msg);
    } finally {
      setDistributing(false);
    }
  };

  const handleConfirm = async () => {
    if (!result) return;
    setConfirming(true);
    try {
      await api.post(`/admin/distribute-classes/${result.distribution_id}/confirm`);
      setConfirmed(true);
      message.success('Distribution confirmed and saved to database.');
    } catch {
      message.error('Failed to confirm distribution.');
    } finally {
      setConfirming(false);
    }
  };

  const classTableColumns = [
    { title: '#', key: 'idx', render: (_: unknown, __: unknown, i: number) => i + 1, width: 50 },
    { title: 'Student ID', dataIndex: 'student_id', key: 'student_id', width: 110 },
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Email', dataIndex: 'email', key: 'email' },
  ];

  const buildTableData = (ids: string[]) =>
    ids.map((id) => ({
      key: id,
      student_id: id,
      name: result?.students_map[id]?.name ?? '—',
      email: result?.students_map[id]?.email ?? '—',
    }));

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        Class Distribution
      </Title>

      {/* ── Configuration Card ───────────────────────────────────────── */}
      <Card style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleDistribute}
          initialValues={{ num_classes: 3 }}
        >
          <Row gutter={16}>
            <Col xs={24} sm={8}>
              <Form.Item
                name="school_year_id"
                label="School Year"
                rules={[{ required: true, message: 'Select a school year' }]}
              >
                <Select
                  placeholder="Select school year"
                  loading={loadingYears}
                  onChange={(v) => fetchClasses(v as number)}
                  options={schoolYears.map((y) => ({ value: y.schoolyearid, label: y.name }))}
                />
              </Form.Item>
            </Col>

            <Col xs={24} sm={8}>
              <Form.Item
                name="class_id"
                label="Grade / Class"
                rules={[{ required: true, message: 'Select a grade' }]}
              >
                <Select
                  placeholder="Select grade"
                  loading={loadingClasses}
                  disabled={classes.length === 0}
                  options={classes.map((c) => ({ value: c.class_id, label: c.name }))}
                />
              </Form.Item>
            </Col>

            <Col xs={24} sm={8}>
              <Form.Item
                name="num_classes"
                label="Number of Classes"
                rules={[{ required: true }]}
              >
                <InputNumber min={2} max={10} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Divider style={{ margin: '8px 0 16px' }} />

          <Row gutter={48}>
            <Col xs={24} md={8}>
              <Form.Item
                label={
                  <Space>
                    <span>Academic Weight</span>
                    <Tag color="blue">{weights.academic}%</Tag>
                  </Space>
                }
              >
                <Slider
                  min={0}
                  max={100}
                  value={weights.academic}
                  onChange={(v) => setWeights((w) => ({ ...w, academic: v }))}
                  trackStyle={{ background: '#4f46e5' }}
                  handleStyle={{ borderColor: '#4f46e5' }}
                />
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label={
                  <Space>
                    <span>Behavior Weight</span>
                    <Tag color="orange">{weights.behavior}%</Tag>
                  </Space>
                }
              >
                <Slider
                  min={0}
                  max={100}
                  value={weights.behavior}
                  onChange={(v) => setWeights((w) => ({ ...w, behavior: v }))}
                  trackStyle={{ background: '#f59e0b' }}
                  handleStyle={{ borderColor: '#f59e0b' }}
                />
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label={
                  <Space>
                    <span>Attendance Weight</span>
                    <Tag color="green">{weights.attendance}%</Tag>
                  </Space>
                }
              >
                <Slider
                  min={0}
                  max={100}
                  value={weights.attendance}
                  onChange={(v) => setWeights((w) => ({ ...w, attendance: v }))}
                  trackStyle={{ background: '#16a34a' }}
                  handleStyle={{ borderColor: '#16a34a' }}
                />
              </Form.Item>
            </Col>
          </Row>

          {weightSum !== 100 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={`Weights sum to ${weightSum}% (not 100%). They will be normalized automatically before distribution.`}
            />
          )}

          <Button
            type="primary"
            htmlType="submit"
            icon={<ExperimentOutlined />}
            loading={distributing}
            size="large"
          >
            Distribute
          </Button>
        </Form>
      </Card>

      {/* ── Results ──────────────────────────────────────────────────── */}
      {distributing && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16, color: '#64748b' }}>Running distribution algorithm…</div>
          </div>
        </Card>
      )}

      {result && !distributing && (
        <Card>
          {/* Balance score header */}
          <Row justify="space-between" align="middle" style={{ marginBottom: 20 }}>
            <Col>
              <div style={{ marginBottom: 4 }}>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  Balance Score
                </Text>
              </div>
              <BalanceBadge
                score={result.balance_metrics.balance_score}
                is_balanced={result.balance_metrics.is_balanced}
              />
              <div style={{ marginTop: 8, color: '#64748b', fontSize: 13 }}>
                {result.balance_metrics.explanation}
              </div>
            </Col>

            <Col>
              <Space>
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => exportCsv(result.classes, result.students_map, 'all')}
                >
                  Export All (CSV)
                </Button>

                {confirmed ? (
                  <Tag color="success" icon={<CheckCircleOutlined />} style={{ padding: '6px 14px', fontSize: 14 }}>
                    Saved to Database
                  </Tag>
                ) : (
                  <Button
                    type="primary"
                    icon={<CheckCircleOutlined />}
                    loading={confirming}
                    onClick={handleConfirm}
                  >
                    Confirm & Save
                  </Button>
                )}
              </Space>
            </Col>
          </Row>

          <Divider style={{ margin: '0 0 16px' }} />

          {/* Per-class tabs */}
          <Tabs
            type="card"
            items={Object.entries(result.classes).map(([className, ids]) => ({
              key: className,
              label: (
                <span>
                  {className}{' '}
                  <Tag style={{ marginLeft: 4 }}>{ids.length}</Tag>
                </span>
              ),
              children: (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
                    <Button
                      size="small"
                      icon={<DownloadOutlined />}
                      onClick={() => exportCsv(result.classes, result.students_map, className)}
                    >
                      Export {className}
                    </Button>
                  </div>
                  <Table
                    size="small"
                    dataSource={buildTableData(ids)}
                    columns={classTableColumns}
                    pagination={{ pageSize: 20, showSizeChanger: false }}
                    rowKey="key"
                  />
                </div>
              ),
            }))}
          />

          <div style={{ marginTop: 12, color: '#94a3b8', fontSize: 13 }}>
            {result.total_students} students distributed across {result.num_classes} classes
          </div>
        </Card>
      )}
    </div>
  );
}
