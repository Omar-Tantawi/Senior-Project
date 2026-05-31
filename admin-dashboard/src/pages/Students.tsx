import { useEffect, useState, useCallback } from 'react';
import {
  Table, Button, Input, Select, Tag, Space, Modal, Form,
  message, Card, Typography, Row, Col, Descriptions, Badge, DatePicker, InputNumber, Tooltip,
} from 'antd';
import {
  PlusOutlined, SearchOutlined, EyeOutlined, EditOutlined, DeleteOutlined,
  SwapOutlined, StopOutlined, CameraOutlined,
} from '@ant-design/icons';
import api from '../api/axios';
import axios from 'axios';
import dayjs from 'dayjs';

const AI_SERVER_URL = 'http://localhost:5000';

const { Title } = Typography;

interface StudentRecord {
  id: number;
  user_id: number;
  date_of_birth: string | null;
  gender: string | null;
  address: string | null;
  enrollment_date: string | null;
  graduation_year: number | null;
  status: string;
  user: {
    id: number;
    name: string;
    email: string;
    phone: string | null;
    is_active: boolean;
  };
  active_enrollment: {
    section: {
      name: string;
      section_id: number;
      school_class: {
        name: string;
        school_year?: { name: string };
      };
    };
  } | null;
}

interface EnrollmentRecord { enrollment_id: number; section_id: number; student_id: number; status: string }
interface Section { section_id: number; name: string; school_class?: { name: string } }

const statusColors: Record<string, string> = {
  active: 'green',
  graduated: 'blue',
  transferred: 'orange',
  withdrawn: 'red',
};

export default function Students() {
  const [data, setData] = useState<{ data: StudentRecord[]; total: number; current_page: number; per_page: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string | undefined>();
  const [page, setPage] = useState(1);

  // Create modal
  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [createForm] = Form.useForm();

  // Edit modal
  const [editOpen, setEditOpen] = useState(false);
  const [editLoading, setEditLoading] = useState(false);
  const [editForm] = Form.useForm();
  const [editingStudent, setEditingStudent] = useState<StudentRecord | null>(null);

  // View modal
  const [viewOpen, setViewOpen] = useState(false);
  const [viewStudent, setViewStudent] = useState<StudentRecord | null>(null);

  // Section assignment modal
  const [sections, setSections] = useState<Section[]>([]);
  const [assignOpen, setAssignOpen] = useState(false);
  const [assignStudent, setAssignStudent] = useState<StudentRecord | null>(null);
  const [assignSectionId, setAssignSectionId] = useState<number | undefined>();
  const [assignLoading, setAssignLoading] = useState(false);

  // Face enrollment
  const [enrollStudent, setEnrollStudent] = useState<StudentRecord | null>(null);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [enrollLoading, setEnrollLoading] = useState(false);

  const fetchStudents = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string | number> = { page, per_page: 15 };
      if (search) params.search = search;
      if (statusFilter) params.status = statusFilter;
      const res = await api.get('/admin/students', { params });
      setData(res.data);
    } catch {
      message.error('Failed to load students');
    } finally {
      setLoading(false);
    }
  }, [page, search, statusFilter]);

  useEffect(() => { fetchStudents(); }, [fetchStudents]);

  useEffect(() => {
    api.get('/admin/sections').then(r => setSections(r.data)).catch(() => {});
  }, []);

  const openAssign = (student: StudentRecord) => {
    setAssignStudent(student);
    setAssignSectionId(undefined);
    setAssignOpen(true);
  };

  const handleAssign = async () => {
    if (!assignStudent || !assignSectionId) return;
    setAssignLoading(true);
    try {
      await api.post('/admin/enrollments', { student_id: assignStudent.id, section_id: assignSectionId, status: 'active' });
      message.success(`${assignStudent.user.name} assigned to section`);
      setAssignOpen(false);
      fetchStudents();
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { message?: string } } };
      message.error(axiosErr.response?.data?.message || 'Failed to assign section');
    } finally {
      setAssignLoading(false);
    }
  };

  const handleRemoveSection = (student: StudentRecord) => {
    const enrollment = student.active_enrollment as unknown as EnrollmentRecord & { enrollment_id: number };
    if (!enrollment) return;
    Modal.confirm({
      title: `Remove ${student.user.name} from their section?`,
      content: `They will be removed from ${student.active_enrollment!.section.school_class.name} — ${student.active_enrollment!.section.name}.`,
      okText: 'Remove',
      okType: 'danger',
      onOk: async () => {
        try {
          // Get enrollment_id via the enrollments endpoint
          const res = await api.get('/admin/enrollments', { params: { student_id: student.id, status: 'active' } });
          const active = res.data[0];
          if (!active) { message.error('No active enrollment found'); return; }
          await api.delete(`/admin/enrollments/${active.enrollment_id}`);
          message.success('Student removed from section');
          fetchStudents();
        } catch {
          message.error('Failed to remove student from section');
        }
      },
    });
  };

  const handleEnrollFace = async () => {
    if (!enrollStudent) return;
    setEnrollLoading(true);
    try {
      const res = await axios.post(`${AI_SERVER_URL}/enroll`, {
        student_id: enrollStudent.id,
        student_name: enrollStudent.user.name,
      }, { timeout: 300000 }); // 5 min timeout — admin captures photos manually

      if (res.data?.success) {
        message.success(res.data.message);
        setEnrollOpen(false);
      } else {
        message.error(res.data?.message || 'Enrollment failed');
      }
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.code === 'ECONNREFUSED') {
        message.error('AI server is not running. Start it with: python main.py --server --port 5000');
      } else {
        message.error('Enrollment failed or was cancelled');
      }
    } finally {
      setEnrollLoading(false);
      setEnrollOpen(false);
    }
  };

  const handleCreate = async (values: Record<string, unknown>) => {
    setCreateLoading(true);
    const payload = {
      ...values,
      date_of_birth: values.date_of_birth ? dayjs(values.date_of_birth as string).format('YYYY-MM-DD') : null,
      enrollment_date: values.enrollment_date ? dayjs(values.enrollment_date as string).format('YYYY-MM-DD') : null,
    };
    try {
      await api.post('/admin/students', payload);
      message.success('Student created successfully');
      setCreateOpen(false);
      createForm.resetFields();
      fetchStudents();
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { message?: string } } };
      message.error(axiosErr.response?.data?.message || 'Failed to create student');
    } finally {
      setCreateLoading(false);
    }
  };

  const handleEdit = async (values: Record<string, unknown>) => {
    if (!editingStudent) return;
    setEditLoading(true);
    const payload = {
      ...values,
      date_of_birth: values.date_of_birth ? dayjs(values.date_of_birth as string).format('YYYY-MM-DD') : null,
    };
    try {
      await api.put(`/admin/students/${editingStudent.id}`, payload);
      message.success('Student updated successfully');
      setEditOpen(false);
      editForm.resetFields();
      setEditingStudent(null);
      fetchStudents();
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { message?: string } } };
      message.error(axiosErr.response?.data?.message || 'Failed to update student');
    } finally {
      setEditLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.delete(`/admin/students/${id}`);
      message.success('Student deleted');
      fetchStudents();
    } catch {
      message.error('Failed to delete student');
    }
  };

  const openEdit = (student: StudentRecord) => {
    setEditingStudent(student);
    editForm.setFieldsValue({
      name: student.user.name,
      email: student.user.email,
      phone: student.user.phone,
      date_of_birth: student.date_of_birth ? dayjs(student.date_of_birth) : null,
      gender: student.gender,
      address: student.address,
      graduation_year: student.graduation_year,
      status: student.status,
    });
    setEditOpen(true);
  };

  const openView = async (id: number) => {
    try {
      const res = await api.get(`/admin/students/${id}`);
      setViewStudent(res.data);
      setViewOpen(true);
    } catch {
      message.error('Failed to load student');
    }
  };

  const columns = [
    {
      title: 'Name',
      key: 'name',
      render: (_: unknown, r: StudentRecord) => r.user.name,
    },
    {
      title: 'Email',
      key: 'email',
      render: (_: unknown, r: StudentRecord) => r.user.email,
    },
    {
      title: 'Class / Section',
      key: 'enrollment',
      render: (_: unknown, r: StudentRecord) => {
        if (!r.active_enrollment) return <Tag>Not enrolled</Tag>;
        const e = r.active_enrollment;
        return `${e.section.school_class.name} — ${e.section.name}`;
      },
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (s: string) => <Tag color={statusColors[s]}>{s}</Tag>,
    },
    {
      title: 'Account',
      key: 'is_active',
      render: (_: unknown, r: StudentRecord) => (
        <Badge status={r.user.is_active ? 'success' : 'error'} text={r.user.is_active ? 'Active' : 'Inactive'} />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: StudentRecord) => (
        <Space size="small">
          <Button size="small" icon={<EyeOutlined />} onClick={() => openView(record.id)} />
          <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(record)} />
          {record.active_enrollment ? (
            <Tooltip title={`Remove from ${record.active_enrollment.section.school_class.name} — ${record.active_enrollment.section.name}`}>
              <Button size="small" icon={<StopOutlined />} danger onClick={() => handleRemoveSection(record)} />
            </Tooltip>
          ) : (
            <Tooltip title="Assign to a section">
              <Button size="small" icon={<SwapOutlined />} onClick={() => openAssign(record)} />
            </Tooltip>
          )}
          <Tooltip title="Enroll face for attendance">
            <Button size="small" icon={<CameraOutlined />} onClick={() => { setEnrollStudent(record); setEnrollOpen(true); }} />
          </Tooltip>
          <Button size="small" icon={<DeleteOutlined />} danger onClick={() => {
            Modal.confirm({
              title: 'Delete this student?',
              content: 'This will permanently delete the student and their user account.',
              okText: 'Delete',
              okType: 'danger',
              onOk: () => handleDelete(record.id),
            });
          }} />
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col><Title level={4} style={{ margin: 0 }}>Students</Title></Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
            Add Student
          </Button>
        </Col>
      </Row>

      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col xs={24} sm={10}>
            <Input
              placeholder="Search by name, email, or phone..."
              prefix={<SearchOutlined />}
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              allowClear
            />
          </Col>
          <Col xs={12} sm={4}>
            <Select
              placeholder="Status"
              style={{ width: '100%' }}
              allowClear
              value={statusFilter}
              onChange={(v) => { setStatusFilter(v); setPage(1); }}
              options={[
                { value: 'active', label: 'Active' },
                { value: 'graduated', label: 'Graduated' },
                { value: 'transferred', label: 'Transferred' },
                { value: 'withdrawn', label: 'Withdrawn' },
              ]}
            />
          </Col>
        </Row>
      </Card>

      <Card>
        <Table
          dataSource={data?.data || []}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            current: data?.current_page || 1,
            total: data?.total || 0,
            pageSize: data?.per_page || 15,
            onChange: (p) => setPage(p),
            showSizeChanger: false,
            showTotal: (total) => `${total} students`,
          }}
        />
      </Card>

      {/* Create Student Modal */}
      <Modal
        title="Add New Student"
        open={createOpen}
        onCancel={() => { setCreateOpen(false); createForm.resetFields(); }}
        footer={null}
        width={500}
      >
        <Form form={createForm} layout="vertical" onFinish={handleCreate}>
          <Form.Item name="name" label="Full Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="email" label="Email" rules={[{ required: true, type: 'email' }]}>
            <Input />
          </Form.Item>
          <Form.Item name="phone" label="Phone">
            <Input />
          </Form.Item>
          <Form.Item name="password" label="Password" rules={[{ required: true, min: 8 }]}>
            <Input.Password />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="gender" label="Gender">
                <Select allowClear options={[
                  { value: 'male', label: 'Male' },
                  { value: 'female', label: 'Female' },
                ]} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="date_of_birth" label="Date of Birth">
                <DatePicker style={{ width: '100%' }} format="DD/MM/YYYY" placeholder="DD/MM/YYYY" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="address" label="Address">
            <Input />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="graduation_year" label="Graduation Year">
                <InputNumber style={{ width: '100%' }} min={2000} max={2100} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="enrollment_date" label="Enrollment Date">
                <DatePicker style={{ width: '100%' }} format="DD/MM/YYYY" placeholder="DD/MM/YYYY" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={createLoading} block>
              Create Student
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Student Modal */}
      <Modal
        title={`Edit Student — ${editingStudent?.user.name}`}
        open={editOpen}
        onCancel={() => { setEditOpen(false); setEditingStudent(null); }}
        footer={null}
        width={500}
      >
        <Form form={editForm} layout="vertical" onFinish={handleEdit}>
          <Form.Item name="name" label="Full Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="email" label="Email" rules={[{ required: true, type: 'email' }]}>
            <Input />
          </Form.Item>
          <Form.Item name="phone" label="Phone">
            <Input />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="gender" label="Gender">
                <Select allowClear options={[
                  { value: 'male', label: 'Male' },
                  { value: 'female', label: 'Female' },
                ]} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="date_of_birth" label="Date of Birth">
                <DatePicker style={{ width: '100%' }} format="DD/MM/YYYY" placeholder="DD/MM/YYYY" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="address" label="Address">
            <Input />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="graduation_year" label="Graduation Year">
                <InputNumber style={{ width: '100%' }} min={2000} max={2100} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="status" label="Status">
                <Select options={[
                  { value: 'active', label: 'Active' },
                  { value: 'graduated', label: 'Graduated' },
                  { value: 'transferred', label: 'Transferred' },
                  { value: 'withdrawn', label: 'Withdrawn' },
                ]} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={editLoading} block>
              Save Changes
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Face Enrollment Modal */}
      <Modal
        title={<Space><CameraOutlined /> Enroll Face — {enrollStudent?.user.name}</Space>}
        open={enrollOpen}
        onCancel={() => { if (!enrollLoading) setEnrollOpen(false); }}
        onOk={handleEnrollFace}
        okText={enrollLoading ? 'Camera is open — press SPACE to capture…' : 'Start Enrollment'}
        confirmLoading={enrollLoading}
        width={440}
      >
        <p>The webcam window will open on the AI server machine.</p>
        <ul style={{ paddingLeft: 20, color: '#555' }}>
          <li>Make sure <strong>{enrollStudent?.user.name}</strong> is in front of the camera</li>
          <li>Press <strong>SPACE</strong> to capture each photo ({enrollStudent ? 5 : 0} photos total)</li>
          <li>Vary the pose slightly between each photo</li>
          <li>Press <strong>Q</strong> to cancel at any time</li>
        </ul>
        <p style={{ color: '#888', fontSize: 12 }}>
          The button will stay loading until all photos are captured.
        </p>
      </Modal>

      {/* Assign Section Modal */}
      <Modal
        title={`Assign Section — ${assignStudent?.user.name}`}
        open={assignOpen}
        onCancel={() => setAssignOpen(false)}
        onOk={handleAssign}
        okText="Assign"
        confirmLoading={assignLoading}
        okButtonProps={{ disabled: !assignSectionId }}
        width={420}
      >
        <p style={{ marginBottom: 12, color: '#666' }}>
          Select a section to enroll this student in. A student can only be in one section at a time.
        </p>
        <Select
          placeholder="Select section"
          style={{ width: '100%' }}
          showSearch
          optionFilterProp="label"
          value={assignSectionId}
          onChange={setAssignSectionId}
          options={sections.map(s => ({
            value: s.section_id,
            label: `${s.school_class?.name || ''} — ${s.name}`,
          }))}
        />
      </Modal>

      {/* View Student Modal */}
      <Modal
        title={`Student Profile — ${viewStudent?.user.name}`}
        open={viewOpen}
        onCancel={() => { setViewOpen(false); setViewStudent(null); }}
        footer={null}
        width={800}
      >
        {viewStudent && (
          <Descriptions column={2} bordered size="small" labelStyle={{ whiteSpace: 'nowrap', width: 130 }}>
            <Descriptions.Item label="Name">{viewStudent.user.name}</Descriptions.Item>
            <Descriptions.Item label="Email">{viewStudent.user.email}</Descriptions.Item>
            <Descriptions.Item label="Phone">{viewStudent.user.phone || '—'}</Descriptions.Item>
            <Descriptions.Item label="Gender">{viewStudent.gender || '—'}</Descriptions.Item>
            <Descriptions.Item label="Date of Birth">{viewStudent.date_of_birth ? dayjs(viewStudent.date_of_birth).format('D MMM YYYY') : '—'}</Descriptions.Item>
            <Descriptions.Item label="Address">{viewStudent.address || '—'}</Descriptions.Item>
            <Descriptions.Item label="Enrollment Date">{viewStudent.enrollment_date ? dayjs(viewStudent.enrollment_date).format('D MMM YYYY') : '—'}</Descriptions.Item>
            <Descriptions.Item label="Graduation Year">{viewStudent.graduation_year || '—'}</Descriptions.Item>
            <Descriptions.Item label="Status">
              <Tag color={statusColors[viewStudent.status]}>{viewStudent.status}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Account">
              <Badge status={viewStudent.user.is_active ? 'success' : 'error'} text={viewStudent.user.is_active ? 'Active' : 'Inactive'} />
            </Descriptions.Item>
            {(viewStudent as unknown as { enrollments?: unknown[] }).enrollments && (
              <Descriptions.Item label="Enrollments" span={2}>
                {(viewStudent as unknown as { enrollments: Array<{ section: { name: string; school_class: { name: string; school_year?: { name: string } } } }> }).enrollments.map((e, i) => (
                  <Tag key={i}>{e.section.school_class.name} — {e.section.name} {e.section.school_class.school_year ? `(${e.section.school_class.school_year.name})` : ''}</Tag>
                ))}
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
}
