<?php

namespace App\Services;

use App\Models\FeePlan;
use App\Models\StudentFeePlan;

class StudentFeePlanService
{
    public function create(array $data): StudentFeePlan
    {
        $exists = StudentFeePlan::where('student_id', $data['student_id'])
            ->where('feeplan_id', $data['feeplan_id'])
            ->exists();

        if ($exists) {
            throw new \RuntimeException('This student is already assigned to this fee plan.');
        }

        $plan  = FeePlan::where('feeplan_id', $data['feeplan_id'])->firstOrFail();
        $paid  = (float) ($data['paid_amount'] ?? 0);
        $total = (float) $plan->totalamount;

        return StudentFeePlan::create([
            'student_id'  => $data['student_id'],
            'feeplan_id'  => $data['feeplan_id'],
            'paid_amount' => $paid,
            'balance'     => max(0, $total - $paid),
            'status'      => $this->resolveStatus($paid, $total),
            'due_date'    => $data['due_date'] ?? null,
            'notes'       => $data['notes'] ?? null,
        ]);
    }

    public function update(StudentFeePlan $account, array $data): void
    {
        if (isset($data['paid_amount'])) {
            $total             = (float) $account->feePlan->totalamount;
            $paid              = (float) $data['paid_amount'];
            $data['balance']   = max(0, $total - $paid);
            $data['status']    = $this->resolveStatus($paid, $total);
        }

        $account->update($data);
    }

    public function delete(StudentFeePlan $account): void
    {
        if ($account->invoices()->exists()) {
            throw new \RuntimeException('Cannot delete: this account has invoices attached.');
        }

        $account->delete();
    }

    private function resolveStatus(float $paid, float $total): string
    {
        if ($paid <= 0)          return 'unpaid';
        if ($paid >= $total)     return 'paid';
        return 'partial';
    }
}
