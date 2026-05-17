<?php

namespace App\Repositories;

use App\Models\Driver;
use App\Models\User;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class DriverRepository
{
    public function filter(array $filters, int $perPage = 15): LengthAwarePaginator
    {
        $query = Driver::with(['user', 'currentBus']);

        if (!empty($filters['search'])) {
            $search = $filters['search'];
            $query->whereHas('user', fn($q) => $q
                ->where('name', 'like', "%{$search}%")
                ->orWhere('email', 'like', "%{$search}%")
                ->orWhere('phone', 'like', "%{$search}%")
            );
        }

        return $query->paginate($perPage);
    }

    public function findWithProfile(int $id): Driver
    {
        return Driver::with(['user', 'assignments.bus'])->findOrFail($id);
    }

    public function createWithUser(array $data): Driver
    {
        return DB::transaction(function () use ($data) {
            $user = User::create([
                'name'      => $data['name'],
                'email'     => $data['email'],
                'phone'     => $data['phone'] ?? null,
                'password'  => Hash::make($data['password']),
                'role_type' => 'driver',
                'is_active' => true,
            ]);

            return Driver::create(['user_id' => $user->id]);
        });
    }

    public function updateWithUser(Driver $driver, array $data): void
    {
        $driver->user->update(array_filter([
            'name'      => $data['name'] ?? null,
            'email'     => $data['email'] ?? null,
            'phone'     => $data['phone'] ?? null,
            'is_active' => $data['is_active'] ?? null,
        ], fn($v) => !is_null($v)));
    }

    public function delete(Driver $driver): void
    {
        DB::transaction(fn() => $driver->user->delete());
    }
}
