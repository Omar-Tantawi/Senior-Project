<?php

namespace App\Repositories;

use App\Models\Student;
use App\Models\User;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class StudentRepository
{
    /**
     * Paginated list with optional filters.
     * Filters: search, status, graduation_year, class_id, section_id
     */
    public function filter(array $filters, int $perPage = 15): LengthAwarePaginator
    {
        $query = Student::with([
            'user',
            'activeEnrollment.section.schoolClass.schoolYear',
        ]);

        if (! empty($filters['search'])) {
            $search = $filters['search'];
            $query->whereHas('user', function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                  ->orWhere('email', 'like', "%{$search}%")
                  ->orWhere('phone', 'like', "%{$search}%");
            });
        }

        if (! empty($filters['status'])) {
            $query->where('status', $filters['status']);
        }

        if (! empty($filters['graduation_year'])) {
            $query->where('graduation_year', $filters['graduation_year']);
        }

        if (! empty($filters['class_id'])) {
            $query->whereHas('enrollments', function ($q) use ($filters) {
                $q->where('status', 'active')
                  ->whereHas('section', function ($q2) use ($filters) {
                      $q2->where('school_class_id', $filters['class_id']);
                  });
            });
        }

        if (! empty($filters['section_id'])) {
            $query->whereHas('enrollments', function ($q) use ($filters) {
                $q->where('status', 'active')
                  ->where('section_id', $filters['section_id']);
            });
        }

        return $query->paginate($perPage);
    }

    /**
     * Find a student with full profile relations for the show endpoint.
     */
    public function findWithProfile(int $id): Student
    {
        return Student::with([
            'user',
            'enrollments.section.schoolClass.schoolYear',
        ])->findOrFail($id);
    }

    /**
     * Find a student with their user record loaded.
     */
    public function findWithUser(int $id): Student
    {
        return Student::with('user')->findOrFail($id);
    }

    /**
     * Create a user + student atomically.
     */
    public function createWithUser(array $data): Student
    {
        return DB::transaction(function () use ($data) {
            $user = User::create([
                'name'      => $data['name'],
                'email'     => $data['email'],
                'phone'     => $data['phone'] ?? null,
                'password'  => Hash::make($data['password']),
                'role_type' => 'student',
                'is_active' => true,
            ]);

            return Student::create([
                'user_id'         => $user->id,
                'date_of_birth'   => $data['date_of_birth'] ?? null,
                'gender'          => $data['gender'] ?? null,
                'address'         => $data['address'] ?? null,
                'enrollment_date' => $data['enrollment_date'] ?? now()->toDateString(),
                'graduation_year' => $data['graduation_year'] ?? null,
                'status'          => $data['status'] ?? 'active',
            ]);
        });
    }

    /**
     * Update user + student fields atomically.
     */
    public function updateWithUser(Student $student, array $data): void
    {
        DB::transaction(function () use ($student, $data) {
            $student->user->update(array_filter([
                'name'      => $data['name'] ?? null,
                'email'     => $data['email'] ?? null,
                'phone'     => $data['phone'] ?? null,
                'is_active' => $data['is_active'] ?? null,
            ], fn ($v) => ! is_null($v)));

            $student->update(array_intersect_key($data, array_flip([
                'date_of_birth',
                'gender',
                'address',
                'enrollment_date',
                'graduation_year',
                'status',
            ])));
        });
    }

    /**
     * Delete a student by deleting their user (FK cascade handles the rest).
     */
    public function delete(Student $student): void
    {
        DB::transaction(fn () => $student->user->delete());
    }
}
