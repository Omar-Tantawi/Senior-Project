<?php

namespace App\Repositories;

use App\Models\Guardian;
use App\Models\StudentGuardian;
use App\Models\User;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class ParentRepository
{
    public function filter(array $filters, int $perPage = 20): LengthAwarePaginator
    {
        $query = Guardian::with(['user', 'studentLinks.student.user']);

        if (!empty($filters['search'])) {
            $search = $filters['search'];
            $query->whereHas('user', fn($q) => $q
                ->where('name', 'ilike', "%{$search}%")
                ->orWhere('email', 'ilike', "%{$search}%")
                ->orWhere('phone', 'ilike', "%{$search}%")
            );
        }

        return $query->paginate($perPage);
    }

    public function findWithProfile(int $id): Guardian
    {
        return Guardian::where('parent_id', $id)
            ->with(['user', 'studentLinks.student.user'])
            ->firstOrFail();
    }

    public function createWithUser(array $data): Guardian
    {
        return DB::transaction(function () use ($data) {
            $user = User::create([
                'name'      => $data['name'],
                'email'     => $data['email'],
                'phone'     => $data['phone'] ?? null,
                'password'  => Hash::make($data['password']),
                'role_type' => 'parent',
                'is_active' => true,
            ]);

            $guardian = Guardian::create(['user_id' => $user->id]);

            foreach ($data['children'] ?? [] as $child) {
                StudentGuardian::create([
                    'student_id'   => $child['student_id'],
                    'parent_id'    => $guardian->parent_id,
                    'relationship' => $child['relationship'],
                    'isprimary'    => $child['isprimary'] ?? false,
                ]);
            }

            return $guardian;
        });
    }

    public function updateWithUser(Guardian $guardian, array $data): void
    {
        $guardian->user->update(array_intersect_key($data, array_flip(['name', 'email', 'phone', 'is_active'])));
    }

    public function delete(Guardian $guardian): void
    {
        User::destroy($guardian->user_id);
    }
}
