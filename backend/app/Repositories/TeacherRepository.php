<?php

namespace App\Repositories;

use App\Models\Teacher;
use App\Models\User;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class TeacherRepository
{
    public function filter(array $filters, int $perPage = 15): LengthAwarePaginator
    {
        $query = Teacher::with(['user', 'subjects']);

        if (!empty($filters['search'])) {
            $search = $filters['search'];
            $query->whereHas('user', fn($q) => $q
                ->where('name', 'like', "%{$search}%")
                ->orWhere('email', 'like', "%{$search}%")
                ->orWhere('phone', 'like', "%{$search}%")
            );
        }

        if (!empty($filters['status'])) {
            $query->where('status', $filters['status']);
        }

        if (!empty($filters['subject_id'])) {
            $query->whereHas('assignments', fn($q) => $q->where('subject_id', $filters['subject_id']));
        }

        return $query->paginate($perPage);
    }

    public function findWithProfile(int $id): Teacher
    {
        return Teacher::with([
            'user',
            'assignments.subject',
            'assignments.section.schoolClass.schoolYear',
        ])->findOrFail($id);
    }

    public function createWithUser(array $data): Teacher
    {
        return DB::transaction(function () use ($data) {
            $user = User::create([
                'name'      => $data['name'],
                'email'     => $data['email'],
                'phone'     => $data['phone'] ?? null,
                'password'  => Hash::make($data['password']),
                'role_type' => 'teacher',
                'is_active' => true,
            ]);

            return Teacher::create([
                'user_id'       => $user->id,
                'date_of_birth' => $data['date_of_birth'] ?? null,
                'gender'        => $data['gender'] ?? null,
                'address'       => $data['address'] ?? null,
                'hire_date'     => $data['hire_date'] ?? now()->toDateString(),
                'status'        => $data['status'] ?? 'active',
            ]);
        });
    }

    public function updateWithUser(Teacher $teacher, array $data): void
    {
        DB::transaction(function () use ($teacher, $data) {
            $teacher->user->update(array_filter([
                'name'      => $data['name'] ?? null,
                'email'     => $data['email'] ?? null,
                'phone'     => $data['phone'] ?? null,
                'is_active' => $data['is_active'] ?? null,
            ], fn($v) => !is_null($v)));

            $teacher->update(array_intersect_key($data, array_flip([
                'date_of_birth', 'gender', 'address', 'hire_date', 'status',
            ])));
        });
    }

    public function delete(Teacher $teacher): void
    {
        DB::transaction(fn() => $teacher->user->delete());
    }
}
